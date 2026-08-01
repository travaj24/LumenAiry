# POP CROSSCHECK, step 6: the three-way picture, per DOE order.
#
#   Zemax POP  |  lumenairy traced-carrier chain  |  exact-ray + RS oracle
#
# on ONE lattice (dx = 0.1 um, +/-20.0 um about each order's own chief ray),
# ONE colour scale shared by every panel of every order, linear AND log10.
#
# WHY THE LOG PANEL IS THE POINT.  Encircled energy at 3/6/12 um is blind to
# where the halo goes -- this campaign has a documented case of a knob priced
# at -0.005 EE3 that returns P/Pin = 1.82.  The log panels are the only place
# a halo shows up as itself.
#
# NORMALISATION, stated rather than assumed:
#   * every panel plots  J = I / (SUM I dA)  over the common window, i.e.
#     irradiance per watt IN THE WINDOW, then divided by ONE global constant
#     (the largest J over all panels) -- so peak differences between orders and
#     between methods are real and directly readable.
#   * "P/Pin" per panel:
#       POP      -- P(window) / P(whole POP array).  POP's array total IS a
#                   real transmission (a forced 8/5/3 mm aperture on surface 20
#                   drops it to 0.947/0.677/0.327), and for this design it
#                   measures 1.00000000 of the launched power at the image
#                   plane for every order -- matching the independent
#                   Gaussian-weighted 70681-ray pupil trace, 0 vignetted.  So
#                   this ratio is a true P/Pin: everything it does not count is
#                   power POP put outside +/-20 um.
#       chain /  -- P(window) relative to the ORACLE's (0,0) window power.
#       oracle      oracle_spot's Rayleigh-Sommerfeld integral omits the
#                   1/(i*lambda) prefactor, so its ABSOLUTE scale is not
#                   calibrated; ratios between its own runs are exact, so
#                   energy is quoted relative to the on-axis oracle.
#   * EE3/EE6/EE12 use focus_scan_121.metrics()'s convention, CHECKED not
#     assumed: r is a RADIUS in microns (3 um radius = 6 um diameter),
#     measured from the PEAK PIXEL, divided by the input power -- here each
#     method's own denominator above.
#
# usage:  python pop_plot_threeway.py
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'pop_profiles')
POPDIR = (r"C:\Users\Tesla\AppData\Local\Temp\claude\C--Users-Tesla"
          r"\372a2d1f-acbe-4b57-a148-eeae3fe1d729\scratchpad\pop121")
ORDERS = [(0, 0), (-1, 0), (-2, 0), (-3, 0), (-4, 0), (-4, -2)]
MRAD_PER_ORDER = 11.5158
DX = 0.1e-6
NAX = 401
LOG_FLOOR = -6.0


def oname(m, n):
    return f"m{m}n{n}".replace('-', 'm')


def load_ours(method, m, n):
    p = os.path.join(OUT, f"{method}_{oname(m, n)}.npz")
    if not os.path.exists(p):
        return None
    d = np.load(p, allow_pickle=True)
    return {'I': np.asarray(d['I'], float), 'ax': np.asarray(d['ax'], float),
            'meta': json.loads(str(d['meta'])), 'src': p}


def load_pop(m, n):
    tag = f"m{m:+d}n{n:+d}".replace('+', 'p').replace('-', 'm')
    hits = sorted(glob.glob(os.path.join(POPDIR, f"prod_{tag}_*.npz")))
    if not hits:
        return None
    d = np.load(hits[-1], allow_pickle=True)
    I = np.asarray(d['I'], float)
    x = np.asarray(d['x'], float) * 1e-3          # mm -> m
    y = np.asarray(d['y'], float) * 1e-3
    meta = json.loads(str(d['meta']))
    # POP pixel is 0.0332 um; block-average by 3 then bilinearly resample onto
    # the common 0.1 um lattice (never up-sampled).
    k = max(1, int(round(DX / abs(np.diff(x).mean()) / 1.0)))
    if k > 1:
        nx = (I.shape[1] // k) * k
        ny = (I.shape[0] // k) * k
        I = I[:ny, :nx].reshape(ny // k, k, nx // k, k).mean(axis=(1, 3))
        x = x[:nx].reshape(-1, k).mean(axis=1)
        y = y[:ny].reshape(-1, k).mean(axis=1)
    ax = (np.arange(NAX) - (NAX - 1) / 2.0) * DX
    fx = np.interp(ax, x, np.arange(x.size), left=np.nan, right=np.nan)
    fy = np.interp(ax, y, np.arange(y.size), left=np.nan, right=np.nan)
    i0 = np.clip(np.floor(fx).astype(int), 0, x.size - 2)
    j0 = np.clip(np.floor(fy).astype(int), 0, y.size - 2)
    tx = np.clip(fx - i0, 0, 1)
    ty = np.clip(fy - j0, 0, 1)
    Ir = ((1 - ty)[:, None] * ((1 - tx)[None, :] * I[np.ix_(j0, i0)]
                               + tx[None, :] * I[np.ix_(j0, i0 + 1)])
          + ty[:, None] * ((1 - tx)[None, :] * I[np.ix_(j0 + 1, i0)]
                           + tx[None, :] * I[np.ix_(j0 + 1, i0 + 1)]))
    Ir = np.nan_to_num(Ir)
    return {'I': Ir, 'ax': ax, 'meta': meta, 'src': hits[-1]}


def metrics_focus_scan(I, ax, denom):
    """focus_scan_121.metrics() convention: r = RADIUS in um from the PEAK
    pixel, EE divided by the input power (``denom``, already an area integral).
    FWHM from the ring-averaged profile with linear interpolation."""
    dx = float(ax[1] - ax[0])
    iy, ix = np.unravel_index(np.argmax(I), I.shape)
    X, Y = np.meshgrid(ax - ax[ix], ax - ax[iy])
    r = np.hypot(X, Y)
    ee = {k: float(I[r <= k * 1e-6].sum()) * dx * dx / denom
          for k in (3, 6, 12)}
    nb = int(min(I.shape[0] // 2, 20e-6 / dx))
    ring = np.clip((r / dx).astype(int), 0, nb)
    s = np.bincount(ring.ravel(), weights=I.ravel(), minlength=nb + 1)
    cn = np.bincount(ring.ravel(), minlength=nb + 1)
    prof = s[:nb] / np.maximum(cn[:nb], 1)
    prof = prof / prof[0]
    idx = np.where(prof < 0.5)[0]
    if len(idx) and idx[0] > 0:
        i1 = idx[0]
        f = (prof[i1 - 1] - 0.5) / (prof[i1 - 1] - prof[i1])
        fwhm = 2 * ((i1 - 1 + 0.5) * dx + f * dx)
    else:
        fwhm = np.nan
    rb = (np.arange(nb) + 0.5) * dx
    cum = np.cumsum(s[:nb]) * dx * dx / denom
    return dict(fwhm=fwhm, ee=ee, prof=prof, rb=rb, cum=cum,
                peak=(float(ax[ix]), float(ax[iy])))


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    data = {}
    for (m, n) in ORDERS:
        data[(m, n)] = {'pop': load_pop(m, n),
                        'chain': load_ours('chain6', m, n),
                        'oracle': load_ours('oracle', m, n)}

    # denominators
    orc00 = data[(0, 0)]['oracle']
    if orc00 is None:
        print("missing oracle (0,0); cannot set the RS energy reference")
        return 2
    den_rs = float(orc00['I'].sum()) * DX * DX

    panels = {}
    gmax = 0.0
    for key in ORDERS:
        for meth in ('pop', 'chain', 'oracle'):
            d = data[key][meth]
            if d is None:
                continue
            I = d['I']
            if meth == 'pop':
                denom = float(d['meta']['P_full']) * 1e-6   # mm^2 -> m^2
            else:
                denom = den_rs
            J = I / denom                       # 1/m^2, per watt of input
            gmax = max(gmax, float(J.max()))
            panels[(key, meth)] = (J, metrics_focus_scan(I, d['ax'], denom),
                                   d)
    print(f"global peak J = {gmax:.4e} /m^2 (per W of input) -> colour scale "
          f"0..1 linear, {LOG_FLOOR}..0 log10, shared by every panel")

    ax_um = (np.arange(NAX) - (NAX - 1) / 2.0) * DX * 1e6
    ext = [ax_um[0], ax_um[-1], ax_um[0], ax_um[-1]]
    written = []
    titles = {'pop': 'Zemax POP  (N=4096, W=0.2 mm)',
              'chain': 'lumenairy chain (6 groups) + exact trailing leg',
              'oracle': 'exact ray + Rayleigh-Sommerfeld oracle'}

    for (m, n) in ORDERS:
        fig, axs = plt.subplots(2, 4, figsize=(21.5, 10.4))
        fig.suptitle(
            f"Design 121, DOE order ({m:+d},{n:+d})   field angle "
            f"({m * MRAD_PER_ORDER:+.2f}, {n * MRAD_PER_ORDER:+.2f}) mrad   "
            f"lambda = 1.31 um\n"
            f"common lattice dx = 0.1 um, +/-20.0 um about each order's own "
            f"chief ray;  COMMON colour scale: J = I/P(window) divided by the "
            f"global peak {gmax:.3e} /m^2/W   (linear 0..1, log10 "
            f"{LOG_FLOOR:.0f}..0)",
            fontsize=12)
        for col, meth in enumerate(('pop', 'chain', 'oracle')):
            got = panels.get(((m, n), meth))
            for row in (0, 1):
                A = axs[row, col]
                if got is None:
                    A.text(0.5, 0.5, 'Zemax unavailable' if meth == 'pop'
                           else 'missing', ha='center', va='center',
                           transform=A.transAxes, fontsize=14, color='crimson')
                    A.set_xticks([])
                    A.set_yticks([])
                    continue
                J, mt, d = got
                Z = J / gmax
                if row == 0:
                    im = A.imshow(Z, extent=ext, origin='lower', vmin=0,
                                  vmax=1.0, cmap='inferno')
                    lbl = 'linear  I/Imax_global'
                else:
                    im = A.imshow(np.log10(np.maximum(Z, 1e-30)), extent=ext,
                                  origin='lower', vmin=LOG_FLOOR, vmax=0.0,
                                  cmap='inferno')
                    lbl = 'log10  I/Imax_global'
                fig.colorbar(im, ax=A, fraction=0.046, pad=0.02)
                for rr, cc in ((3.0, 'cyan'), (6.0, 'lime')):
                    A.add_patch(Circle((mt['peak'][0] * 1e6,
                                        mt['peak'][1] * 1e6), rr, fill=False,
                                       ec=cc, lw=1.1, ls='--'))
                if meth == 'pop':
                    pw = float(J.sum()) * (DX * DX)     # = P(win)/P(array)
                    ptxt = f"P(win)/P(POP array) = {pw * 100:.3f} %"
                else:
                    pw = float(J.sum()) * (DX * DX)
                    ptxt = f"P(win) rel. oracle(0,0) = {pw * 100:.3f} %"
                A.set_title(f"{titles[meth]}\n{lbl}", fontsize=9)
                A.set_xlabel('x - chief ray  [um]', fontsize=8)
                A.set_ylabel('y - chief ray  [um]', fontsize=8)
                A.text(0.02, 0.98,
                       f"FWHM {mt['fwhm'] * 1e6:.3f} um\n"
                       f"EE3  {mt['ee'][3] * 100:6.2f} %\n"
                       f"EE6  {mt['ee'][6] * 100:6.2f} %\n"
                       f"EE12 {mt['ee'][12] * 100:6.2f} %\n{ptxt}",
                       transform=A.transAxes, va='top', ha='left', fontsize=8,
                       color='white', family='monospace',
                       bbox=dict(fc='black', alpha=0.55, ec='none'))
        # radial profile + encircled energy
        Ap, Ae = axs[0, 3], axs[1, 3]
        for meth, col in (('pop', 'tab:red'), ('chain', 'tab:blue'),
                          ('oracle', 'k')):
            got = panels.get(((m, n), meth))
            if got is None:
                continue
            _J, mt, _d = got
            Ap.semilogy(mt['rb'] * 1e6, np.maximum(mt['prof'], 1e-8),
                        color=col, label=meth)
            Ae.plot(mt['rb'] * 1e6, mt['cum'] * 100, color=col, label=meth)
        for A, ttl, yl in ((Ap, 'ring-averaged radial profile (log)',
                            'I(r)/I(0)'),
                           (Ae, 'encircled energy about the peak',
                            'EE(r)  [% of input]')):
            A.set_title(ttl, fontsize=9)
            A.set_xlabel('radius [um]', fontsize=8)
            A.set_ylabel(yl, fontsize=8)
            A.grid(alpha=0.3)
            A.legend(fontsize=8)
            for rr, cc in ((3.0, 'c'), (6.0, 'g'), (12.0, '0.5')):
                A.axvline(rr, color=cc, ls='--', lw=0.9)
        Ap.set_ylim(1e-8, 2)
        Ae.set_ylim(0, 105)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fn = os.path.join(OUT, f"pop_threeway_order_{oname(m, n)}.png")
        fig.savefig(fn, dpi=110)
        plt.close(fig)
        written.append(fn)
        print("wrote", fn)

    # per-method overview strips, log10, all orders on the shared scale
    for meth in ('pop', 'chain', 'oracle'):
        fig, axs = plt.subplots(2, len(ORDERS), figsize=(4.0 * len(ORDERS), 9))
        fig.suptitle(f"{titles[meth]} -- all orders, shared colour scale "
                     f"(linear top, log10 bottom; global peak {gmax:.3e} "
                     f"/m^2/W)", fontsize=13)
        for j, key in enumerate(ORDERS):
            got = panels.get((key, meth))
            for row in (0, 1):
                A = axs[row, j]
                if got is None:
                    A.text(0.5, 0.5, 'n/a', ha='center', va='center',
                           transform=A.transAxes, color='crimson')
                    A.set_xticks([])
                    A.set_yticks([])
                    continue
                J, mt, _d = got
                Z = J / gmax
                if row == 0:
                    A.imshow(Z, extent=ext, origin='lower', vmin=0, vmax=1,
                             cmap='inferno')
                else:
                    A.imshow(np.log10(np.maximum(Z, 1e-30)), extent=ext,
                             origin='lower', vmin=LOG_FLOOR, vmax=0,
                             cmap='inferno')
                A.set_title(f"({key[0]:+d},{key[1]:+d})  EE3 "
                            f"{mt['ee'][3] * 100:.2f} %", fontsize=9)
                for rr, cc in ((3.0, 'cyan'), (6.0, 'lime')):
                    A.add_patch(Circle((mt['peak'][0] * 1e6,
                                        mt['peak'][1] * 1e6), rr, fill=False,
                                       ec=cc, lw=1.0, ls='--'))
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fn = os.path.join(OUT, f"pop_allorders_{meth}.png")
        fig.savefig(fn, dpi=105)
        plt.close(fig)
        written.append(fn)
        print("wrote", fn)

    # raw grids for POP, on the common lattice, next to the figures
    for (m, n) in ORDERS:
        got = panels.get(((m, n), 'pop'))
        if got is None:
            continue
        J, mt, d = got
        fn = os.path.join(OUT, f"pop_{oname(m, n)}.npz")
        np.savez_compressed(fn, I=d['I'],
                            ax=(np.arange(NAX) - (NAX - 1) / 2.0) * DX,
                            meta=json.dumps({**d['meta'], 'm': m, 'n': n,
                                             'resampled_to_common_dx': DX,
                                             'source': d['src']}))
        written.append(fn)

    print("\n=== TABLE: focus_scan_121.metrics() convention "
          "(r = RADIUS in um, from the peak pixel) ===")
    print("order      angle[mrad]     method   FWHM[um]   EE3     EE6    EE12"
          "    P(win)/norm")
    for (m, n) in ORDERS:
        for meth in ('pop', 'chain', 'oracle'):
            got = panels.get(((m, n), meth))
            if got is None:
                print(f"({m:+d},{n:+d})  ---   {meth:8s} MISSING")
                continue
            J, mt, _d = got
            pw = float(J.sum()) * DX * DX
            print(f"({m:+d},{n:+d})  ({m * MRAD_PER_ORDER:+7.2f},"
                  f"{n * MRAD_PER_ORDER:+7.2f})  {meth:7s} "
                  f"{mt['fwhm'] * 1e6:8.3f} {mt['ee'][3] * 100:7.2f} "
                  f"{mt['ee'][6] * 100:7.2f} {mt['ee'][12] * 100:7.2f}"
                  f"    {pw * 100:8.3f} %")
    print(f"\n{len(written)} files written to {OUT}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
