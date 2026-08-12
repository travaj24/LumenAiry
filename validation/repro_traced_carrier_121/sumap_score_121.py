# SUM-AT-APERTURE probe -- scoring.  Reads what sumap_probe_121.py wrote and
# prints every table the audit note quotes.  No propagation happens here.
#
#   python sumap_score_121.py                 # all tables
#   python sumap_score_121.py --json out.json
#
# The per-frame spot block is fan_multi_121.py's OWN (ring-averaged radial
# profile, in-frame EE radii, argmax peak), reproduced line for line so the
# numbers are directly comparable to the campaign's banners and bars:
#   EE bar 0.1 point, energy bar 4e-5, FWHM to the printed digits.
import argparse
import glob
import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
import sys  # noqa: E402

sys.path.insert(0, _HERE)
import sumap_probe_121 as P  # noqa: E402

DXO = P.DXO
ORDERS = P.ORDERS


def spot(F, dxo=DXO):
    """fan_multi_121.py's per-frame spot metrics, verbatim in behaviour."""
    F = np.asarray(F)
    NT = F.shape[-1]
    I = np.abs(F) ** 2
    tot = float(I.sum())
    iy, ix = np.unravel_index(np.argmax(I), I.shape)
    rr = np.hypot((np.arange(I.shape[1]) - ix)[None, :] * dxo,
                  (np.arange(I.shape[0]) - iy)[:, None] * dxo)
    nb = min(NT // 2, 400)
    ring = np.clip((rr / dxo).astype(int), 0, nb)
    s = np.bincount(ring.ravel(), weights=I.ravel(), minlength=nb + 1)
    cn = np.bincount(ring.ravel(), minlength=nb + 1)
    prof = (s[:nb] / np.maximum(cn[:nb], 1)) / I[iy, ix]
    rb = (np.arange(nb) + 0.5) * dxo
    idx = np.where(prof < 0.5)[0]
    axx = (np.arange(I.shape[1]) - NT / 2) * dxo
    axy = (np.arange(I.shape[0]) - NT / 2) * dxo
    return dict(
        fwhm=float(2 * rb[idx[0]]) if len(idx) else float('nan'),
        ee3=float(I[rr <= 3e-6].sum()) / tot,
        ee6=float(I[rr <= 6e-6].sum()) / tot,
        ee12=float(I[rr <= 12e-6].sum()) / tot,
        power=tot * dxo * dxo, peak=float(I[iy, ix]),
        peak_x=float(axx[ix]), peak_y=float(axy[iy]),
        cx=float((I.sum(axis=0) * axx).sum() / tot),
        cy=float((I.sum(axis=1) * axy).sum() / tot))


def compare(A, B):
    """Field-level arm-A vs arm-B comparison, including the piston/fringe
    check: a chain-to-chain piston shows up as ``piston`` here, and an
    INCONSISTENT one shows up as a spread of ``piston`` across frames."""
    A = np.asarray(A)
    B = np.asarray(B)
    nA = float(np.linalg.norm(A))
    ip = complex(np.vdot(A, B))               # sum conj(A) * B
    piston = float(np.angle(ip))
    rel = float(np.linalg.norm(B - A) / nA)
    rel_p = float(np.linalg.norm(B - np.exp(1j * piston) * A) / nA)
    # amplitude-weighted phase residual over the bright core only
    a = np.abs(A)
    msk = a > 0.1 * a.max()
    d = np.angle(B[msk] * np.conj(A[msk])) - piston
    d = (d + np.pi) % (2 * np.pi) - np.pi
    w = a[msk] ** 2
    return dict(rel_l2=rel, rel_l2_after_piston=rel_p, piston=piston,
                phase_rms_core=float(np.sqrt((w * d * d).sum() / w.sum())),
                overlap=float(abs(ip) / (nA * float(np.linalg.norm(B)))),
                amp_ratio=float(np.linalg.norm(B) / nA))


def load_arm_b(path):
    d = np.load(path, allow_pickle=False)
    info = json.loads(str(d['info']))
    frames = {o: d[f'f_{P._tag(*o)}'] for o in ORDERS}
    return frames, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', default=os.path.join(_HERE,
                                                   '_sumap_score.json'))
    ap.add_argument('--glob', default='_sumap_B*.npz')
    a = ap.parse_args()
    out = {'orders': [list(o) for o in ORDERS], 'arms': {}}

    # ---- ARM A -------------------------------------------------------------
    A, metaA = {}, {}
    for o in ORDERS:
        metaA[o], A[o] = P.load_meta(*o)
    print("ARM A -- shipped per-order path (one exact leg per order)")
    print(f"{'order':>9s} {'wall_s':>8s} {'retrace':>8s} {'readout':>8s} "
          f"{'FWHM_um':>8s} {'EE3_%':>7s} {'EE6_%':>7s} {'EE12_%':>7s} "
          f"{'power':>12s} {'peak(x,y)um':>16s}")
    sA = {}
    for o in ORDERS:
        s = spot(A[o])
        sA[o] = s
        m = metaA[o]
        print(f"{str(o):>9s} {m['wall']:8.1f} {m['t_retrace']:8.1f} "
              f"{m['t_readout']:8.1f} {s['fwhm'] * 1e6:8.3f} "
              f"{s['ee3'] * 100:7.3f} {s['ee6'] * 100:7.3f} "
              f"{s['ee12'] * 100:7.3f} {s['power']:12.6e} "
              f"({s['peak_x'] * 1e6:+6.2f},{s['peak_y'] * 1e6:+6.2f})")
    out['arms']['A'] = {str(o): dict(spot=sA[o], meta=metaA[o])
                        for o in ORDERS}

    # ---- ARM B variants ----------------------------------------------------
    for path in sorted(glob.glob(os.path.join(_HERE, a.glob))):
        frames, info = load_arm_b(path)
        summed = [tuple(x) for x in info['summed']]
        name = os.path.basename(path)
        single = (len(summed) == 1)
        print()
        print(f"ARM B [{name}]  summed orders {summed}   "
              f"dx_c {info['dx_c'] * 1e6:.4f} um  N_c {info['n_c']}  "
              f"(window {info['n_c'] * info['dx_c'] * 1e3:.4f} mm)")
        print(f"  t_reref {info['t_reref']:.1f}s  t_resample "
              f"{info['t_resample']:.1f}s")
        print(f"{'frame':>9s} {'FWHM_um':>8s} {'EE3_%':>7s} {'EE6_%':>7s} "
              f"{'EE12_%':>7s} {'power':>12s} {'P/P_A':>10s} {'relL2':>10s} "
              f"{'relL2_pist':>10s} {'piston':>9s} {'ph_rms':>8s}")
        rows = {}
        for o in ORDERS:
            s = spot(frames[o])
            c = compare(A[o], frames[o])
            rows[str(o)] = dict(spot=s, cmp=c)
            print(f"{str(o):>9s} {s['fwhm'] * 1e6:8.3f} {s['ee3'] * 100:7.3f} "
                  f"{s['ee6'] * 100:7.3f} {s['ee12'] * 100:7.3f} "
                  f"{s['power']:12.6e} {s['power'] / sA[o]['power']:10.7f} "
                  f"{c['rel_l2']:10.3e} {c['rel_l2_after_piston']:10.3e} "
                  f"{c['piston']:+9.5f} {c['phase_rms_core']:8.2e}")
        if not single:
            pis = [rows[str(o)]['cmp']['piston'] for o in ORDERS]
            print(f"  INTER-FRAME PISTON: {['%+.6f' % p for p in pis]}  "
                  f"spread {max(pis) - min(pis):+.3e} rad")
        out['arms'][name] = dict(info={k: v for k, v in info.items()
                                       if k != 'rows'}, frames=rows,
                                 summed=[list(o) for o in summed])

    # ---- CROSSTALK, from the single-order arm-B runs ------------------------
    # The crosstalk and linearity tables must be built from runs on the SAME
    # common grid AND the same leg; a pitch ablation is a different experiment
    # and is separated here by its grid, not by its filename.
    def _grid(info):
        return (int(info['n_c']), round(float(info['dx_c']), 15))

    grids = sorted({_grid(load_arm_b(p)[1])
                    for p in glob.glob(os.path.join(_HERE, a.glob))})
    for mode, gk in [(m, g) for g in grids for m in ('full', 'crop')]:
        sing = {}
        for path in sorted(glob.glob(os.path.join(_HERE, a.glob))):
            frames, info = load_arm_b(path)
            if info.get('leg_mode', 'full') != mode or _grid(info) != gk:
                continue
            s = [tuple(x) for x in info['summed']]
            if len(s) == 1:
                sing[s[0]] = frames
        if len(sing) != len(ORDERS):
            continue
        print()
        print(f"ARM B CROSSTALK [leg={mode}, N_c={gk[0]}, dx_c="
              f"{gk[1] * 1e6:.4f} um] -- power in frame i from a SINGLE-ORDER "
              f"sum j")
        print("  (arm A cannot show this at all: its frame i is order i's own")
        print("   tile and nothing else, so its off-diagonal is 0 by")
        print("   construction, not by physics.)")
        hdr = ''.join(f"{str(j):>16s}" for j in ORDERS)
        print(f"{'frame i \\ order j':>20s}{hdr}")
        Xt = {}
        for i in ORDERS:
            cells, line = [], f"{str(i):>20s}"
            for j in ORDERS:
                p = float((np.abs(sing[j][i]) ** 2).sum()) * DXO * DXO
                cells.append(p)
                line += f"{p:16.6e}"
            print(line)
            d = cells[ORDERS.index(i)]
            Xt[str(i)] = dict(power=cells,
                              rel_to_diag=[c / d for c in cells])
        print(f"{'ratio to diagonal':>20s}")
        for i in ORDERS:
            line = f"{str(i):>20s}"
            for v in Xt[str(i)]['rel_to_diag']:
                line += f"{v:16.3e}"
            print(line)
        out.setdefault('crosstalk', {})[f'{mode}_{gk[0]}'] = Xt
        # linearity: does the summed run equal the sum of the singles?
        for path in sorted(glob.glob(os.path.join(_HERE, a.glob))):
            frames, info = load_arm_b(path)
            if info.get('leg_mode', 'full') != mode or _grid(info) != gk:
                continue
            s = [tuple(x) for x in info['summed']]
            if len(s) <= 1:
                continue
            print()
            print(f"ARM B LINEARITY [{os.path.basename(path)}]: "
                  f"|| sum_j B_j - B_all || / || B_all ||")
            lin = {}
            for i in ORDERS:
                tot = sum(sing[j][i] for j in s)
                r = float(np.linalg.norm(tot - frames[i])
                          / np.linalg.norm(frames[i]))
                lin[str(i)] = r
                print(f"   frame {str(i):>9s}: {r:.3e}")
            out.setdefault('linearity', {})[os.path.basename(path)] = lin
    with open(a.json, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {a.json}")


if __name__ == '__main__':
    main()
