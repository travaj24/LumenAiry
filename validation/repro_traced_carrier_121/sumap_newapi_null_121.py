# NULL CONTROL of PROBE_SUM_AT_APERTURE_2026_08_11, re-run through the NEW
# lumenairy.propagators.carrier_field API.
#
# WHAT THIS IS FOR.  The probe measured arm B with harness-side code
# (``sumap_probe_121.py``).  ``carrier_field.py`` productizes that code path.
# The claim under test is not "the new API is plausible" but "the new API
# REPRODUCES THE PROBE'S OWN NUMBERS", so this script re-runs the probe's
# section-5 null control -- ONE order through re-reference + resample + the
# like-for-like ``crop`` leg, against the shipped per-order tile -- with every
# aggregation step replaced by the library primitives.
#
# WHAT IT REUSES AND WHY THAT IS THE STRONG FORM.  It consumes the probe's own
# cached artifacts:
#
#   _sumap_ap_<tag>_nfc8192.npy   the BYTE-IDENTICAL back-aperture field arm A
#                                 propagated (the probe's read-only spy on
#                                 _fine_trace_group_exit captured it)
#   _sumap_A_<tag>_nfc8192.npz    arm A's weighted tile + the full metadata of
#                                 the readout call that produced it
#
# so nothing upstream of the seam can differ between this run and the probe's,
# and no chain is re-run (which also means the section-1.2 hazard -- a library
# edit moving the chain's absolute phase -- cannot contaminate the comparison:
# the aperture field is a file, not a computation).
#
# The caches live in the ORIGINAL working tree and are read READ-ONLY; point
# SUMAP_CACHE at them.  Nothing is written there.
#
#   python sumap_newapi_null_121.py [--orders 0,0;-2,0;-4,-2] [--json out.json]
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

# ``python <script>.py`` puts the SCRIPT's directory on sys.path, not the cwd,
# so an unrelated pip-installed lumenairy would win over the checkout this
# script belongs to.  Pin the checkout explicitly, in front.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import lumenairy  # noqa: E402
import lumenairy.propagators.carrier as CAR  # noqa: E402
from lumenairy.propagators.carrier_field import (  # noqa: E402
    CarrierField,
    CarrierSpec,
    FieldGrid,
    aggregate,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
# The probe's cache directory.  Defaults to this one; override when the
# artifacts live in another checkout of the same repo.
CACHE = os.environ.get(
    'SUMAP_CACHE',
    r'D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics\Lumenairy'
    r'\validation\repro_traced_carrier_121')

# -- the probe's configuration of record (PROBE_SUM_AT_APERTURE S1 / S3) ----
LAM = 1.31e-6
TRAILING = 7.7058e-3
ORDERS = ((0, 0), (-2, 0), (-4, -2))
DXO = 0.2e-6
TILE = 1024
NFC = 8192
WF = 4.0
DXC = 1.2292e-6          # the pitch of record (S3; pitch-converged in S8.1)
NC = 8192
RAMB = float('inf')

# The probe's own section-5 crop-leg null-control results, for the
# reproduction check.  These are the numbers the audit note prints.
PROBE_NULL = {
    (0, 0):   dict(rel_l2=2.7785e-05, piston=+7.287e-09, ph_rms=1.84e-06,
                   fwhm=3.400e-6, ee3=0.907407, power=1.8727654e-09),
    (-2, 0):  dict(rel_l2=1.4026e-04, piston=-7.716e-08, ph_rms=2.33e-05,
                   fwhm=3.400e-6, ee3=0.906343, power=1.8735537e-09),
    (-4, -2): dict(rel_l2=9.3424e-05, piston=-1.741e-08, ph_rms=5.31e-06,
                   fwhm=3.800e-6, ee3=0.900711, power=1.8794228e-09),
}


def _tag(m, n):
    return f"{m:+d}_{n:+d}".replace('+', 'p').replace('-', 'm')


def _ap_path(m, n):
    return os.path.join(CACHE, f'_sumap_ap_{_tag(m, n)}_nfc{NFC}.npy')


def _meta_path(m, n):
    return os.path.join(CACHE, f'_sumap_A_{_tag(m, n)}_nfc{NFC}.npz')


def load_meta(m, n):
    d = np.load(_meta_path(m, n), allow_pickle=False)
    return json.loads(str(d['meta'])), d['tile']


def spot(F, dxo=DXO):
    """sumap_score_121.spot, verbatim in behaviour (which is itself
    fan_multi_121.py's own per-frame spot block)."""
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
    return dict(fwhm=float(2 * rb[idx[0]]) if len(idx) else float('nan'),
                ee3=float(I[rr <= 3e-6].sum()) / tot,
                ee6=float(I[rr <= 6e-6].sum()) / tot,
                ee12=float(I[rr <= 12e-6].sum()) / tot,
                power=tot * dxo * dxo, peak=float(I[iy, ix]))


def compare(A, B):
    """sumap_score_121.compare, verbatim in behaviour."""
    A = np.asarray(A)
    B = np.asarray(B)
    nA = float(np.linalg.norm(A))
    ip = complex(np.vdot(A, B))
    piston = float(np.angle(ip))
    a = np.abs(A)
    msk = a > 0.1 * a.max()
    d = np.angle(B[msk] * np.conj(A[msk])) - piston
    d = (d + np.pi) % (2 * np.pi) - np.pi
    w = a[msk] ** 2
    return dict(rel_l2=float(np.linalg.norm(B - A) / nA), piston=piston,
                phase_rms_core=float(np.sqrt((w * d * d).sum() / w.sum())),
                amp_ratio=float(np.linalg.norm(B) / nA))


def arm_a_readout_grid(m, n):
    """sumap_probe_121.arm_a_readout_grid, verbatim: the crop window and
    internal fine grid arm A's OWN readout used, recomputed from arm A's own
    back-aperture field with the readout's own sizing arithmetic."""
    M_ = load_meta(m, n)[0]
    E = np.load(_ap_path(m, n), mmap_mode='r')
    dx = float(M_['dx_ap'])
    cen = (M_['chief_exit'][0] - M_['grid_origin'][0],
           M_['chief_exit'][1] - M_['grid_origin'][1])
    w_amp = float(CAR._envelope_amp_radius(np.asarray(E), dx, dx, centre=cen))
    del E
    N = int(M_['n_ap'])
    dec = bool(cen[0] or cen[1] or M_['tilt_exit'][0] or M_['tilt_exit'][1])
    avail = N * dx - 2.0 * max(abs(cen[0]), abs(cen[1])) if dec else N * dx
    win = min(WF * w_amp, avail)
    n_crop = int(min(max(int(2 * round((win / dx) / 2)), 2), N))
    win = n_crop * dx
    na = min(max(w_amp / abs(float(M_['R_out'])), 0.02), 0.95)
    dxf = LAM / (3.0 * na)
    n_fine = int(2 ** int(np.ceil(np.log2(max(win / dxf, n_crop)))))
    n_fine = min(n_fine, NFC)
    return dict(w_amp=w_amp, win=win, n_crop=n_crop, n_fine=n_fine,
                dx_fine=win / n_fine)


def null_control_one(m, n, verbose=True):
    """ONE order through the NEW API, then the probe's own crop leg."""
    M_, tileA = load_meta(m, n)
    w = complex(M_['weight'][0], M_['weight'][1])
    dx_k = float(M_['dx_ap'])
    origin = (M_['grid_origin'][0], M_['grid_origin'][1])
    chief = (M_['chief_exit'][0], M_['chief_exit'][1])
    tilt = (M_['tilt_exit'][0], M_['tilt_exit'][1])
    R_k = float(M_['R_out'])
    R_c = float(load_meta(0, 0)[0]['R_out'])

    grid_k = FieldGrid((int(M_['n_ap']), int(M_['n_ap'])), dx_k, origin=origin)
    car_k = CarrierSpec(R=R_k, centre=chief, tilt=tilt, piston=0.0)
    grid_c = FieldGrid((NC, NC), DXC, origin=(0.0, 0.0))
    car_c = CarrierSpec(R=R_c, centre=(0.0, 0.0), tilt=(0.0, 0.0), piston=0.0)

    t0 = time.perf_counter()
    E = np.load(_ap_path(m, n))
    t_load = time.perf_counter() - t0

    # (1) PRIMITIVE 1 -- wrap the aperture field as a CarrierField (divides
    #     the order's own congruence out, on its own grid frame).
    t0 = time.perf_counter()
    f_k = CarrierField.from_full_field(
        E, grid_k, car_k, LAM,
        provenance={'order': [m, n], 'source': 'sumap arm-A back aperture'})
    del E
    t_wrap = time.perf_counter() - t0

    # (2) PRIMITIVE 2 -- re-reference onto the common carrier / grid, with
    #     the order's own Dammann weight, through ``aggregate`` so the
    #     energy ledger is exercised on the real geometry.
    t0 = time.perf_counter()
    res = aggregate([f_k], car_c, grid_c, weights=[w], on_window='warn')
    del f_k
    t_reref = time.perf_counter() - t0
    row = res.ledger.rows[0]

    # (3) the probe's own LIKE-FOR-LIKE crop leg on the reconstructed field.
    t0 = time.perf_counter()
    acc = res.field.full_field()
    g = arm_a_readout_grid(m, n)
    w_sum = float(CAR._envelope_amp_radius(acc, DXC, DXC, centre=tuple(chief)))
    frame = CAR.carrier_referenced_exact_focus_readout(
        acc, R_c, TRAILING, LAM, DXC, dx_out=DXO, N_out=TILE,
        tilt=(0.0, 0.0), centre_out=tuple(M_['centre_out']), ram_budget=RAMB,
        on_ram_cap='error', on_replica='warn', on_readout_window='warn',
        on_n_fine_cap='error',
        N_fine=g['n_fine'], window_factor=g['win'] / w_sum,
        centre=tuple(chief))
    del acc
    t_leg = time.perf_counter() - t0

    sA, sB = spot(tileA), spot(frame)
    c = compare(tileA, frame)
    out = dict(
        order=[m, n], armA=sA, armB=sB, cmp=c,
        power_ratio=sB['power'] / sA['power'],
        ledger=dict(power_source=row.power_source,
                    power_on_common=row.power_on_common,
                    frac_out_of_window=row.frac_out_of_window,
                    support_radius=row.support_radius,
                    containment_margin=row.containment_margin,
                    nyquist_margin=row.nyquist_margin,
                    binding_term=row.binding_term,
                    resampled=row.resampled),
        nyquist=res.field.provenance.get('aggregate_ledger', {}),
        timing=dict(load=t_load, wrap=t_wrap, reref=t_reref, leg=t_leg),
        leg=dict(win=g['win'], n_fine=g['n_fine'], dx_fine=g['dx_fine'],
                 w_armA=g['w_amp'], w_sum=w_sum))
    if verbose:
        p = PROBE_NULL[(m, n)]
        print(f"({m:+d},{n:+d})  NEW API   relL2 {c['rel_l2']:.4e}   "
              f"piston {c['piston']:+.3e}   ph_rms "
              f"{c['phase_rms_core']:.2e}   FWHM {sB['fwhm'] * 1e6:.3f} um   "
              f"EE3 {sB['ee3'] * 100:.4f}   power {sB['power']:.7e}")
        print(f"          PROBE     relL2 {p['rel_l2']:.4e}   "
              f"piston {p['piston']:+.3e}   ph_rms {p['ph_rms']:.2e}   "
              f"FWHM {p['fwhm'] * 1e6:.3f} um   EE3 {p['ee3'] * 100:.4f}   "
              f"power {p['power']:.7e}")
        print(f"          P/P_A {sB['power'] / sA['power']:.7f}   "
              f"ledger: support {row.support_radius * 1e3:.4f} mm, "
              f"out-of-window {row.frac_out_of_window:.3e}, "
              f"nyquist margin {row.nyquist_margin:.3f}x "
              f"({row.binding_term}), containment "
              f"{row.containment_margin * 1e3:+.4f} mm")
        print(f"          leg win {g['win'] * 1e3:.6f} mm  n_fine "
              f"{g['n_fine']}  dx_fine {g['dx_fine'] * 1e6:.10f} um    "
              f"[load {t_load:.1f}s  wrap {t_wrap:.1f}s  reref "
              f"{t_reref:.1f}s  leg {t_leg:.1f}s]", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--orders', default=None)
    ap.add_argument('--json', default=os.path.join(
        _HERE, '_sumap_newapi_null.json'))
    a = ap.parse_args()
    ords = (ORDERS if not a.orders else
            tuple(tuple(int(v) for v in p.split(','))
                  for p in a.orders.split(';') if p.strip()))
    print("NULL CONTROL through lumenairy.propagators.carrier_field")
    print(f"  library {lumenairy.__version__} @ {lumenairy.__file__}")
    print(f"  cache   {CACHE}")
    print(f"  common  dx_c {DXC * 1e6:.4f} um  N_c {NC}  "
          f"(window {NC * DXC * 1e3:.4f} mm)")
    rows = []
    for (m, n) in ords:
        if not (os.path.exists(_ap_path(m, n))
                and os.path.exists(_meta_path(m, n))):
            raise SystemExit(
                f"({m},{n}): the probe's cached aperture field / arm-A tile "
                f"is missing under {CACHE}.  Rebuild with "
                f"`python sumap_probe_121.py arma` in that tree, or point "
                f"SUMAP_CACHE at the tree that has them.")
        rows.append(null_control_one(m, n))
    with open(a.json, 'w') as fh:
        json.dump(rows, fh, indent=1)
    print(f"\nwrote {a.json}")


if __name__ == '__main__':
    main()
