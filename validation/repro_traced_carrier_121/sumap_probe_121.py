# SUM-AT-APERTURE probe for the design-121 traced chain (2026-08-11).
#
# LOCAL-ONLY.  NO library edit: everything here is harness-side, and every
# apply-side operation is the LIBRARY's own function, imported and called.
#
# THE HYPOTHESIS UNDER TEST
# -------------------------
#   Run each DOE order's chain only as far as the LAST GROUP'S BACK APERTURE,
#   re-reference every order onto ONE common analytic carrier, sum the
#   envelopes on one common grid, run ONE exact final leg on the sum, and read
#   each frame with its own Bluestein zoom.  By linearity that should equal the
#   shipped per-order path, at ~1/32 of the final-leg cost.
#
# THE SEAM, AND WHY IT IS WHERE IT IS  (read this before the numbers)
# -------------------------------------------------------------------
# ``propagate_traced_carrier_chain`` CAN be stopped at the last group's exit
# vertex with ``final_distance=0`` and no ``focus_readout`` -- the chain-A
# pattern -- but the field it returns there is on the CO-MOVING COARSE grid
# (design 121: dx ~ 33 um) and the exit sphere at NA 0.405 needs dx <= 1.6 um.
# That is the entire reason the exact final leg exists: it re-traces the last
# group onto a grid that Nyquist-samples its exit sphere
# (``_fine_trace_group_exit``) and only then propagates.  A sum taken on the
# coarse exit plane cannot be propagated exactly, and a sum taken BEFORE the
# last group cannot be traced at all (``apply_real_lens_traced``'s
# entrance->exit map is single-congruence by construction -- the reason the
# per-order fan architecture exists).
#
# So the ONE plane where the orders can legitimately be summed is the last
# group's back aperture ON THE EXACT LEG'S FINE RETRACE GRID.  That plane is
# internal to ``propagate_traced_carrier_chain``, so this probe reaches it with
# a read-only SPY on ``_fine_trace_group_exit`` -- and arm B then consumes the
# BYTE-IDENTICAL array arm A propagated, which is the strongest available
# isolation: any arm-A/arm-B difference is owned by re-reference / resample /
# sum / leg and by nothing upstream of them.
#
# CONSEQUENCE FOR THE COST CLAIM, stated up front: the per-order fine RETRACE
# is upstream of the seam and is NOT saved.  The architecture can only ever
# save the exact READOUT (sphere reference + upsample + band-limited ASM
# Bluestein zoom).  This probe measures both halves separately so the ceiling
# is a measurement, not an assumption.
#
# SUBCOMMANDS
#   census   -- geometry only (delegates to sumap_census_121)
#   arma     -- ARM A: the shipped per-order path, one exact leg per order.
#               Caches each order's tile AND its back-aperture field.
#   armb     -- ARM B: re-reference + resample + sum + ONE leg + per-frame
#               Bluestein zooms.  ``--orders`` restricts the SUM (the null
#               control and the crosstalk runs use that).
#   score    -- compare, and write the tables.
#
# ENV / FLAGS: NFC (n_fine_cap, default 8192), DXC (common pitch, m),
# NC (common grid size), NW (Newton workers, default 1 -- CAPSTONE_D121 sec 5).
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings('ignore', message='.*prescription aperture.*')
warnings.filterwarnings('ignore', message='.*residual transverse.*')
warnings.filterwarnings('ignore', message='.*under-sampled.*')

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import _d121_common as C  # noqa: E402

import lumenairy as la  # noqa: E402
import lumenairy.propagators.carrier as CAR  # noqa: E402
from lumenairy.propagators.mft import (  # noqa: E402
    angular_spectrum_propagate_mft,
)

# ---- configuration (the shipped fan's own, at the affordable NFC) ----------
ORDERS = ((0, 0), (-2, 0), (-4, -2))
RS = 4
NW = int(os.environ.get('NW', '1'))
DXO = 0.2e-6                 # common image-plane pitch (fan_multi_121 default)
TILE = 1024                  # per-frame readout window (fan_multi_121 default)
NFC = int(os.environ.get('NFC', '8192'))
WF = 4.0
LEG = 'auto'                 # resolves EXACT at 121's na_exit = 0.405
RAMB = float('inf')          # explicit, so no clamp can degrade the grid
N_OUT_FAN = 32768            # the full-fan common lattice; used ONLY to snap
                             # tile centres -- never allocated.


def _peak_gb():
    """Peak working set + peak commit of THIS process, in GB.  Windows keeps
    both counters itself, so this is the true peak rather than a sample."""
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return (float(getattr(mi, 'peak_wset', mi.rss)) / 1e9,
                float(getattr(mi, 'peak_pagefile', 0.0)) / 1e9)
    except Exception:
        return (float('nan'), float('nan'))


def _tag(m, n):
    return f"{m:+d}_{n:+d}".replace('+', 'p').replace('-', 'm')


def _ap_path(m, n):
    return os.path.join(_HERE, f'_sumap_ap_{_tag(m, n)}_nfc{NFC}.npy')


def _meta_path(m, n):
    return os.path.join(_HERE, f'_sumap_A_{_tag(m, n)}_nfc{NFC}.npz')


def setup():
    """Everything order-independent: groups, order table, chain-A field."""
    _pre, groups_post, _gap, period = C.geometry()
    mx, my, amps = C.order_table(period)
    env_doe, R_doe, dx_doe, P_in = C.chain_a()
    tab = {}
    for i in range(len(mx)):
        tab[(int(mx[i]), int(my[i]))] = complex(amps[i])
    return dict(groups_post=groups_post, period=period, table=tab,
                env_doe=env_doe, R_doe=R_doe, dx_doe=dx_doe, P_in=P_in)


def frame_centre(S, m, n):
    """The readout window centre for order (m,n): the analytic chief ray at
    the image plane, SNAPPED to the common output lattice -- exactly what
    ``propagate_traced_carrier_chain_multi._window`` does (both arms use it)."""
    car = la.TiltedCarrier(float(S['R_doe']), m * C.LAM / S['period'],
                           n * C.LAM / S['period'])
    x, y, _L, _M = CAR._chain_chief_ray_at_target(
        S['groups_post'], C.LAM, car, C.TRAILING, 'sumap_probe_121')
    return (round(x / DXO) * DXO, round(y / DXO) * DXO), (x, y), car


# ===========================================================================
# ARM A -- the shipped path, with a read-only spy at the back aperture
# ===========================================================================
def arm_a_one(S, m, n, force=False):
    ap, meta = _ap_path(m, n), _meta_path(m, n)
    if os.path.exists(ap) and os.path.exists(meta) and not force:
        print(f"  ({m:+d},{n:+d}) cached", flush=True)
        return
    (cx, cy), (xr, yr), car = frame_centre(S, m, n)
    fr = dict(dx_out=DXO, N_out=TILE, centre_out=(cx, cy),
              n_fine_cap=NFC, window_factor=WF, ram_budget=RAMB)
    cap = {}
    o_ftge = CAR._fine_trace_group_exit
    o_ro = CAR.carrier_referenced_exact_focus_readout

    def _ftge(*a, **kw):
        t0 = time.perf_counter()
        out = o_ftge(*a, **kw)
        cap['t_retrace'] = time.perf_counter() - t0
        cap['E_ap'], cap['dx_ap'] = out[0], float(out[1])
        cap['R_out'] = float(a[8])
        cap['grid_origin'] = tuple(
            float(v) for v in kw['grid_origin_out'].get('origin', (0.0, 0.0)))
        return out

    def _ro(E_full, R, z, wl, dx, **kw):
        t0 = time.perf_counter()
        out = o_ro(E_full, R, z, wl, dx, **kw)
        cap['t_readout'] = time.perf_counter() - t0
        cap['ro'] = dict(R=float(R), z=float(z), dx=float(dx),
                         centre=tuple(float(v) for v in kw.get('centre',
                                                               (0.0, 0.0))),
                         tilt=tuple(float(v) for v in kw.get('tilt',
                                                             (0.0, 0.0))),
                         centre_out=tuple(float(v) for v in
                                          kw.get('centre_out', (0.0, 0.0))),
                         window_factor=float(kw.get('window_factor', 7.0)),
                         n_fine_cap=int(kw.get('n_fine_cap', 0)))
        return out

    CAR._fine_trace_group_exit = _ftge
    CAR.carrier_referenced_exact_focus_readout = _ro
    t0 = time.perf_counter()
    try:
        res = CAR.propagate_traced_carrier_chain(
            S['env_doe'], S['groups_post'], C.LAM, S['dx_doe'], r_in=car,
            ray_subsample=RS, n_workers=NW, final_distance=C.TRAILING,
            focus_readout=fr, final_leg=LEG)
    finally:
        CAR._fine_trace_group_exit = o_ftge
        CAR.carrier_referenced_exact_focus_readout = o_ro
    wall = time.perf_counter() - t0
    w = S['table'][(m, n)]
    tile = np.asarray(res.field) * w
    E_ap = np.asarray(cap['E_ap'])
    # GRID OF RECORD, asserted rather than assumed (ADJUDICATION sec 2.1).
    if E_ap.shape[-1] != NFC:
        raise SystemExit(f"({m},{n}): the retrace leg ran at n_fine="
                         f"{E_ap.shape[-1]}, not the requested {NFC}.  The "
                         f"RAM clamp degraded the grid of record; refusing.")
    np.save(_ap_path(m, n), E_ap)
    st = [s for s in res.stages if s.get('exact_final') and 'na_exit' in s]
    tgt = [s for s in res.stages if s.get('target')]
    np.savez_compressed(
        _meta_path(m, n), tile=tile, weight=np.array(w),
        meta=np.array(json.dumps(dict(
            order=[m, n], weight=[w.real, w.imag], dx_ap=cap['dx_ap'],
            n_ap=int(E_ap.shape[-1]), R_out=cap['R_out'],
            grid_origin=list(cap['grid_origin']), ro=cap['ro'],
            centre_out=[cx, cy], chief_pred=[xr, yr],
            chief_exit=[float(st[0].get('x_c_out', 0.0)),
                        float(st[0].get('y_c_out', 0.0))] if st else [0.0, 0.0],
            tilt_exit=[float(st[0].get('L_out', 0.0)),
                       float(st[0].get('M_out', 0.0))] if st else [0.0, 0.0],
            na_exit=float(st[0].get('na_exit', float('nan'))) if st else None,
            na_exit_measured=float(st[0].get('na_exit_measured',
                                             float('nan'))) if st else None,
            power_exit=float(st[0].get('power', float('nan'))) if st else None,
            readout_period=list(tgt[0]['readout_period']) if tgt else None,
            t_retrace=cap['t_retrace'], t_readout=cap['t_readout'],
            peak_wset_gb=_peak_gb()[0], peak_commit_gb=_peak_gb()[1],
            wall=wall, nfc=NFC, dxo=DXO, tile=TILE, nw=NW))))
    print(f"  ({m:+d},{n:+d}) wall {wall:7.1f}s  retrace {cap['t_retrace']:6.1f}s"
          f"  readout {cap['t_readout']:6.1f}s  dx_ap {cap['dx_ap'] * 1e6:.4f} um"
          f"  origin ({cap['grid_origin'][0] * 1e3:+.4f},"
          f"{cap['grid_origin'][1] * 1e3:+.4f}) mm", flush=True)


def arm_a(force=False, orders=ORDERS):
    S = setup()
    print(f"ARM A -- shipped per-order path, NFC={NFC}, NW={NW}")
    for (m, n) in orders:
        arm_a_one(S, m, n, force=force)
    return S


def seam_check(order=(0, 0)):
    """MEASURE the seam claim instead of asserting it: run the chain-A
    ``final_distance`` pattern (stop at the last group's exit vertex, no
    focus_readout) and compare the pitch it lands on with the pitch the exit
    sphere needs.  Cheap -- there is no exact leg on this path."""
    S = setup()
    m, n = order
    car = la.TiltedCarrier(float(S['R_doe']), m * C.LAM / S['period'],
                           n * C.LAM / S['period'])
    t0 = time.perf_counter()
    res = CAR.propagate_traced_carrier_chain(
        S['env_doe'], S['groups_post'], C.LAM, S['dx_doe'], r_in=car,
        ray_subsample=RS, n_workers=NW, final_distance=0.0)
    wall = time.perf_counter() - t0
    st = res.stages[-1]
    R_out = float(st['R_out'])
    w = float(st['w'])
    na = w / abs(R_out)
    need = C.LAM / (2.0 * na)
    mA = load_meta(*order)[0] if os.path.exists(_meta_path(*order)) else None
    print(f"SEAM CHECK -- chain-A pattern, order ({m:+d},{n:+d}), "
          f"final_distance=0, no focus_readout   [{wall:.1f} s]")
    print(f"  returns dx = {float(res.dx) * 1e6:.4f} um on a "
          f"{np.shape(res.field)[-1]}-square grid, R_out = {R_out * 1e3:.6f} mm")
    print(f"  co-moving envelope radius w = {w * 1e3:.4f} mm  ->  exit NA "
          f"{na:.4f}  ->  the exit sphere needs dx <= {need * 1e6:.4f} um")
    print(f"  UNDER-SAMPLED BY {float(res.dx) / need:.1f}x  -- this plane "
          f"cannot carry the exit congruence, which is why the exact leg "
          f"re-traces the last group.")
    if mA is not None:
        print(f"  the exact leg's own retrace grid: dx = "
              f"{mA['dx_ap'] * 1e6:.4f} um on {mA['n_ap']}-square "
              f"({float(res.dx) / mA['dx_ap']:.1f}x finer), which is the plane "
              f"this probe sums on.")
    return dict(dx_coarse=float(res.dx), n_coarse=int(np.shape(res.field)[-1]),
                R_out=R_out, w=w, na=na, dx_needed=need, wall=wall)


def load_meta(m, n):
    d = np.load(_meta_path(m, n), allow_pickle=False)
    return json.loads(str(d['meta'])), d['tile']


# ===========================================================================
# ARM B -- re-reference onto ONE carrier, resample, sum, ONE leg
# ===========================================================================
def _congruence_phase(shape, dx, R, chief, tilt, sign):
    """The order's OWN analytic carrier on a grid, as the library builds it:
    exact sphere about the chief ray, plus the tilt ramp, plus the niche-C5
    exactness term.  ``sign=-1`` divides it out, ``+1`` restores it.  These
    are ``carrier_referenced_exact_focus_readout``'s own three lines, called
    on the same library functions."""
    cx, cy = float(chief[0]), float(chief[1])
    L, M = float(tilt[0]), float(tilt[1])
    k = 2.0 * np.pi / C.LAM
    S_ = CAR._exact_sphere_eikonal(shape, dx, dx, C.LAM, R, centre=(cx, cy))
    ph = np.exp((sign * 1j * k) * S_)
    del S_
    rp = CAR._tilt_ramp(shape, dx, C.LAM, L, M, cx, cy, sign)
    if rp is not None:
        ph *= rp
        del rp
    xf = CAR._tilt_exactness_phase(shape, dx, dx, C.LAM, R, L, M, sign,
                                   centre=(cx, cy))
    if xf is not None:
        ph *= xf
    return ph


def arm_a_readout_grid(m, n):
    """The crop window and internal fine grid ARM A's own readout used for
    this order, recomputed from arm A's own back-aperture field with
    ``carrier_referenced_exact_focus_readout``'s own sizing arithmetic.  Used
    to give the ARM-B 'crop' leg a LIKE-FOR-LIKE transform (same physical
    window, same fine pitch, same transform sizes) instead of a bigger one."""
    M_ = load_meta(m, n)[0]
    E = np.load(_ap_path(m, n), mmap_mode='r')
    dx = float(M_['dx_ap'])
    cen = (M_['chief_exit'][0] - M_['grid_origin'][0],
           M_['chief_exit'][1] - M_['grid_origin'][1])
    w_amp = float(CAR._envelope_amp_radius(np.asarray(E), dx, dx, centre=cen))
    N = int(M_['n_ap'])
    dec = bool(cen[0] or cen[1] or M_['tilt_exit'][0] or M_['tilt_exit'][1])
    avail = N * dx - 2.0 * max(abs(cen[0]), abs(cen[1])) if dec else N * dx
    win = min(WF * w_amp, avail)
    n_crop = int(min(max(int(2 * round((win / dx) / 2)), 2), N))
    win = n_crop * dx
    na = min(max(w_amp / abs(float(M_['R_out'])), 0.02), 0.95)
    dxf = C.LAM / (3.0 * na)
    n_fine = int(2 ** int(np.ceil(np.log2(max(win / dxf, n_crop)))))
    n_fine = min(n_fine, NFC)
    return dict(w_amp=w_amp, win=win, n_crop=n_crop, n_fine=n_fine,
                dx_fine=win / n_fine)


def arm_b(orders=ORDERS, dx_c=None, n_c=None, verbose=True, keep_sum=False,
          leg_modes=('full',)):
    """Returns ``(frames, info)``: ``frames[(m,n)]`` is the TILE-sized field on
    the same window arm A used, for every order in ORDERS (not just the summed
    ones -- that is how crosstalk is read)."""
    S = setup()
    metas = {o: load_meta(*o)[0] for o in orders}
    # COMMON GRID (sumap_census_121 + the measured aperture beam):
    #   pitch   -- the binding Nyquist at this plane is the UNION band, the
    #              inter-order carrier-offset ramp (0.2776 rad) PLUS each
    #              beam's own exit NA (0.532 measured): dx <= 0.8089 um.  The
    #              ramp-only bound the protocol names is 2.3593 um, so the
    #              0.6146 um default is 3.84x inside it and 1.32x inside the
    #              union bound.  DXC=0.4045e-6 with NC=21504 is the 2.0x-union
    #              ablation.
    #   size    -- the beams sit 0.96 mm apart at this plane (they are
    #              separated SPATIALLY here, not angularly) and each carries
    #              99.999 % of its power inside 2.64 mm, so a 10.07 mm common
    #              window clears the most displaced order by 3.1 mm.
    dx_c = float(dx_c if dx_c is not None
                 else os.environ.get('DXC', 0.6146e-6))
    n_c = int(n_c if n_c is not None else os.environ.get('NC', 16384))
    # the COMMON carrier: the central frame's own exit congruence -- same R
    # (the paraxial closure is order-independent), chief ray on axis, no tilt.
    m0 = load_meta(0, 0)[0]
    R_c = float(m0['R_out'])
    t_ref = t_res = 0.0
    acc = np.zeros((n_c, n_c), dtype=np.complex128)
    rows = []
    for (m, n) in orders:
        M_ = metas[(m, n)]
        dx_k = float(M_['dx_ap'])
        gox, goy = M_['grid_origin']
        chief = M_['chief_exit']
        tilt = M_['tilt_exit']
        w = complex(M_['weight'][0], M_['weight'][1])
        E = np.load(_ap_path(m, n))
        t0 = time.perf_counter()
        # (1) divide out the order's OWN congruence, in ITS OWN grid frame
        env = E * _congruence_phase(E.shape, dx_k, float(M_['R_out']),
                                    (chief[0] - gox, chief[1] - goy), tilt, -1)
        del E
        t_ref += time.perf_counter() - t0
        # (2) band-limited resample onto the COMMON lattice (absolute
        #     coordinates, centred on the common carrier's chief ray).  z=0
        #     ASM-MFT == the library's own exact band-limited interpolation;
        #     the separable Bluestein keeps the working set at (N x L).
        t0 = time.perf_counter()
        env_c = angular_spectrum_propagate_mft(
            env, 0.0, C.LAM, dx_k, dx_c, n_c,
            centre_out=(-gox, -goy), bandlimit=False,
            _bluestein_separable=True)
        del env
        t_res += time.perf_counter() - t0
        # (3) restore the order's congruence on the common grid, in ABSOLUTE
        #     coordinates -- this is the re-reference: what survives is the
        #     analytic difference between the order's carrier and the common
        #     one, which is never fitted, only evaluated.
        t0 = time.perf_counter()
        env_c *= _congruence_phase((n_c, n_c), dx_c, float(M_['R_out']),
                                   chief, tilt, +1)
        acc += w * env_c
        p_k = float((np.abs(env_c) ** 2).sum()) * dx_c * dx_c * abs(w) ** 2
        del env_c
        t_ref += time.perf_counter() - t0
        rows.append(dict(order=[m, n], dx_ap=dx_k, power_on_common=p_k,
                         power_exit=M_['power_exit']))
        if verbose:
            print(f"  summed ({m:+d},{n:+d}): power on the common grid "
                  f"{p_k:.6e} vs chain exit {M_['power_exit'] * abs(w) ** 2:.6e}"
                  f"  ratio {p_k / (M_['power_exit'] * abs(w) ** 2):.9f}",
                  flush=True)
    # (4) ONE exact final leg on the SUM + one Bluestein zoom per frame.
    #     window_factor is set so the crop is the WHOLE common grid and
    #     N_fine == n_c, i.e. the readout's internal resample is an identity
    #     and the leg runs at exactly the pitch chosen above.
    out = {}
    for leg_mode in leg_modes:
      frames, t_leg, legs = {}, {}, {}
      for (m, n) in ORDERS:
        (cx, cy), _pred, _car = frame_centre(S, m, n)
        if leg_mode == 'full':
            # THE ARCHITECTURE AS SPECIFIED: one leg on the whole summed
            # aperture field, at the common pitch, per frame a Bluestein zoom.
            kw = dict(N_fine=n_c, window_factor=1e6, centre=(0.0, 0.0))
            legs[(m, n)] = dict(win=n_c * dx_c, n_fine=n_c, dx_fine=dx_c)
        else:
            # LIKE-FOR-LIKE variant: crop the SUM about THIS frame's chief ray
            # to the same physical window arm A used, and run the same fine
            # grid.  The carrier referenced is still the ONE common sphere --
            # only its centre moves to this frame's chief ray, which is legal
            # precisely because every order shares R at this plane.
            g = arm_a_readout_grid(m, n)
            ch = load_meta(m, n)[0]['chief_exit']
            w_sum = float(CAR._envelope_amp_radius(acc, dx_c, dx_c,
                                                   centre=tuple(ch)))
            kw = dict(N_fine=g['n_fine'], window_factor=g['win'] / w_sum,
                      centre=tuple(ch))
            legs[(m, n)] = dict(win=g['win'], n_fine=g['n_fine'],
                                dx_fine=g['dx_fine'], w_sum=w_sum,
                                w_armA=g['w_amp'])
        t0 = time.perf_counter()
        f = CAR.carrier_referenced_exact_focus_readout(
            acc, R_c, C.TRAILING, C.LAM, dx_c, dx_out=DXO, N_out=TILE,
            tilt=(0.0, 0.0), centre_out=(cx, cy), ram_budget=RAMB,
            on_ram_cap='error', on_replica='warn', on_readout_window='warn',
            on_n_fine_cap='error', **kw)
        t_leg[(m, n)] = time.perf_counter() - t0
        frames[(m, n)] = f
        if verbose:
            print(f"  [{leg_mode}] frame ({m:+d},{n:+d}) zoom "
                  f"{t_leg[(m, n)]:.1f}s", flush=True)
      pk = _peak_gb()
      out[leg_mode] = (frames, dict(
          # WHICH LIBRARY.  The working tree was edited by another process
          # DURING this probe (sec 2.1 of the audit note): the chain's absolute
          # output phase moves with it while every intensity metric holds, so
          # an arm-A/arm-B pair is only meaningful within ONE source hash.
          lib_sha=C._lumenairy_source_sha()[:16],
          dx_c=dx_c, n_c=n_c, R_c=R_c, rows=rows, t_reref=t_ref,
          leg_mode=leg_mode, legs={str(k): v for k, v in legs.items()},
          t_resample=t_res, t_leg={str(k): v for k, v in t_leg.items()},
          t_leg_total=float(sum(t_leg.values())),
          peak_wset_gb=pk[0], peak_commit_gb=pk[1],
          summed=[list(o) for o in orders]))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('census', 'arma', 'armb', 'seam'))
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--orders', default=None,
                    help='m,n;m,n -- restrict the ARM-B SUM')
    ap.add_argument('--out', default=None)
    ap.add_argument('--suffix', default='')
    ap.add_argument('--leg', default='full',
                    choices=('full', 'crop', 'both'))
    a = ap.parse_args()
    if a.cmd == 'census':
        import sumap_census_121
        sumap_census_121.census()
    elif a.cmd == 'seam':
        seam_check()
    elif a.cmd == 'arma':
        arm_a(force=a.force)
    elif a.cmd == 'armb':
        ords = (ORDERS if not a.orders else
                tuple(tuple(int(v) for v in p.split(','))
                      for p in a.orders.split(';') if p.strip()))
        modes = ('full', 'crop') if a.leg == 'both' else (a.leg,)
        res = arm_b(orders=ords, leg_modes=modes)
        for mode, (fr, info) in res.items():
            out = os.path.join(
                _HERE, f'_sumap_B{mode[0]}_'
                + '_'.join(_tag(*o) for o in ords) + a.suffix + '.npz')
            np.savez_compressed(
                out, info=np.array(json.dumps(info)),
                **{f'f_{_tag(*o)}': fr[o] for o in ORDERS})
            print('wrote', out, flush=True)
