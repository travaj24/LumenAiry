# Design-121 chain + THROUGH-FOCUS scan (the acceptance runner): every prior
# metric in this campaign was taken exactly at the fixed MSoP plane
# (final_distance = TRAILING); with NA 0.152 the focused Rayleigh range is
# ~18 um, so focus-position errors of tens of um confound cross-config
# comparisons.  This runner executes the chain, then scans the readout field
# through +/-DZ um (plain fixed-grid Bluestein re-propagation of the readout
# plane) and reports at-plane AND best-focus metrics.
#
# DEFAULTS = THE SHIPPING CONFIGURATION.  With CREF / AM / PIP unset the
# script passes NO chain kwargs at all, so it measures whatever
# propagate_traced_carrier_chain defaults to -- since v5.29 the validated
# carrier-regime triple (carrier_reference='sphere' +
# amplitude_model='ray_density' + preserve_input_phase='remap').  Expected
# at N=2048 / NFC=8192 / WF=4.0, best focus -> FWHM 3.450 um / EE3 88.8 /
# EE6 99.6 / EE12 99.8, ON-AXIS -- the post-S12 SHIPPING-DEFAULT acceptance,
# equal to the measured ideal-field ceiling for this readout
# (audit AUDIT_TRACED_PRODUCTION_READINESS_2026_07_24; the earlier
# 3.550 / 88.4 / 99.3 line was the pre-S12 configuration).
#
# Pre-v5.29 this script HARD-CODED AM='ray_density' + PIP='0' as its env
# DEFAULTS, so a plain run silently measured the intermediate `rd+pip0`
# plateau (best focus ~5.65 um / 46.6 / 76.7) instead of the shipping
# configuration -- exactly the "harness measures a stale config" failure the
# S6.8 candidate-2 false negative came from (audit S8.4).  The knobs are now
# opt-IN overrides:
#   CREF=sphere|parabola   -> carrier_reference=
#   AM=ray_density|screen  -> traced_kwargs['amplitude_model']
#   PIP=0|1|remap          -> traced_kwargs['preserve_input_phase']
# e.g. the legacy pre-flip chain is CREF=parabola AM=screen PIP=1.
# PowerShell reminder: `$env:AM=''` UNSETS the variable.
#   RAMB=auto|inf|<GB>     -> the fine grid's RAM-budget INTENT.  The NFC this
#                             runner asks for is PROVEN reachable before the
#                             chain starts and asserted afterwards; a box that
#                             cannot hold it exits 2 instead of quietly
#                             reporting a spot measured on half the grid.
#
# IMPORT-SAFE: the body is behind ``if __name__ == '__main__':``.  This runner
# passes n_workers=8, and ``_lens_traced._script_has_main_guard`` forces the
# Newton pool SERIAL from an unguarded script -- so without the guard the knob
# was inert, and under any spawn pool the whole chain would have re-run in
# every child.
import ast, os, re, sys, time, warnings
import numpy as np

# NEVER blanket-silence.  This line was ``warnings.filterwarnings('ignore')``
# until 2026-08-10, with no comment saying what it was for, and it SWALLOWED
# the RAM clamp's "RESOLUTION-LIMITED (non-converged)" message -- so a run
# whose fine grid had been halved printed the same shaped AT-PLANE /
# BEST-FOCUS banner as one that had not, and there was no way to tell them
# apart from the output (ADJUDICATION_NFC_8192_2026_08_10.md sec 0, the
# silent-8192 finding; reproduced directly in
# FIX_PERF_PARALLEL_2026_08_10.md sec 2).
#
# The three ignores below are the campaign's OWN documented suppressions,
# copied verbatim from fan_multi_121.py, which never blanket-silenced and
# records why each exists: they are the known-noisy, understood categories on
# this design.  Everything else -- the clamp, the Nyquist guards, the
# replica/clipping guards, the Newton non-convergence counts -- comes through.
warnings.filterwarnings('ignore', message='.*prescription aperture.*')
warnings.filterwarnings('ignore', message='.*residual transverse.*')
warnings.filterwarnings('ignore', message='.*under-sampled.*')
sys.path.insert(0, r"D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics\Lumenairy")
sys.path.insert(0, r"D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics\Reverse_Symmetric_ASM")
import lumenairy as la
from lumenairy import GLASS_REGISTRY, GLASS_VALIDITY, SELLMEIER_COEFFICIENTS
from lumenairy.propagators.mft import angular_spectrum_propagate_mft

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _grid_intent as gi        # noqa: E402  (import-safe: no module body work)

RUNNER = (r"D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics"
          r"\Reverse_Symmetric_ASM\run_poc_119_120_v518.py")
ZMX = (r"D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics"
       r"\Reverse_Symmetric_ASM\tx4designstudy121\20260707 dll Tx02-MSOP16.zmx")
lam = 1.31e-6
w0 = 4e-6
OBJ_GAP = 47.906284e-3
TRAILING = 7.7058e-3

# --- IMPORT-SAFE BOUNDARY --------------------------------------------------
# Everything with a side effect sits behind this guard: a spawn worker
# re-imports __main__, and the Newton pool this runner asks for (n_workers=8)
# is forced serial by _lens_traced._script_has_main_guard while the module is
# unguarded.  The library backstop stays; this is what makes the knob real.
if __name__ == '__main__':

    src = open(RUNNER, 'r', encoding='utf-8', errors='ignore').read()
    m = re.search(r'_NEW_GLASSES\s*=\s*\{', src)
    i0 = m.end() - 1
    d = 0
    for i in range(i0, len(src)):
        if src[i] == '{': d += 1
        elif src[i] == '}':
            d -= 1
            if d == 0: break
    for g, (c, r, nd) in ast.literal_eval(src[i0:i+1]).items():
        if g not in SELLMEIER_COEFFICIENTS:
            SELLMEIER_COEFFICIENTS[g] = c; GLASS_REGISTRY[g] = '__sellmeier__'; GLASS_VALIDITY[g] = r

    import tx_design_study_sim as sim
    rx = la.load_zemax_zmx(ZMX)
    events = sim.group_elements_into_lenses(rx)
    z1 = 2e-3
    gapsum = 0.0
    first = True
    groups = []
    for ev in events:
        if ev['type'] != 'lens':
            gapsum += ev['z_before']
            continue
        g = (OBJ_GAP - z1) if first else (ev['z_before'] + gapsum)
        gapsum = 0.0
        groups.append({'prescription': ev['prescription'], 'gap_before': g})
        first = False

    N = int(os.environ.get('RN', '2048'))
    RS = int(os.environ.get('RS', '4'))
    NFC = int(os.environ.get('NFC', '8192'))
    WF = float(os.environ.get('WF', '4.0'))
    NOUT = int(os.environ.get('NOUT', '2048'))
    AM = os.environ.get('AM', '')
    PIP = os.environ.get('PIP', '')
    CREF = os.environ.get('CREF', '')
    _tkw = {}
    if AM:
        _tkw['amplitude_model'] = AM
    if PIP:
        # Unrecognised values must NOT fall through as "library default" -- that is
        # how the S6.8 candidate-2 false negative happened (a harness that only
        # honoured PIP=0 silently ran PIP=True for every other value).
        if PIP not in ('0', '1', 'remap'):
            raise SystemExit(f"PIP must be '0', '1', 'remap' or unset, got {PIP!r}")
        _tkw['preserve_input_phase'] = {'0': False, '1': True,
                                        'remap': 'remap'}[PIP]
    _ckw = {}
    if CREF:
        if CREF not in ('sphere', 'parabola'):
            raise SystemExit(
                f"CREF must be 'sphere', 'parabola' or unset, got {CREF!r}")
        _ckw['carrier_reference'] = CREF
    _fr = {'dx_out': 0.05e-6, 'N_out': NOUT, 'n_fine_cap': NFC,
           'window_factor': WF}
    zR = np.pi * w0 * w0 / lam
    w_z1 = w0 * np.sqrt(1 + (z1 / zR) ** 2)
    R1 = z1 * (1 + (zR / z1) ** 2)
    dx0 = float(os.environ.get('DX0', 1.0e-6 * 2048 / N))
    x = (np.arange(N) - N // 2) * dx0
    env0 = np.exp(-(x[None, :] ** 2 + x[:, None] ** 2) / w_z1 ** 2
                  ).astype(np.complex128)
    P_in = float(np.sum(np.abs(env0) ** 2)) * dx0 * dx0

    # The NFC this runner scores at is PROVEN before the chain runs, not read
    # out of a warning afterwards.  See _grid_intent.py for why.
    _RB = gi.preflight(NFC, os.environ.get('RAMB', 'auto'), workers=1,
                       n_px=N * N, n_out=NOUT,
                       label=f"design-121 focus scan, NFC={NFC}")
    if _RB is not None:
        _fr['ram_budget'] = _RB
    t0 = time.time()
    print(f"config: N={N} rs={RS} nfc={NFC} wf={WF} nout={NOUT} "
          f"chain_kwargs={ {**_ckw, **({'traced_kwargs': _tkw} if _tkw else {})} }"
          + ("  (pure library defaults)" if not (_ckw or _tkw) else ""))
    with gi.record_warnings() as _wrec:
        res = la.propagate_traced_carrier_chain(
            env0, groups, lam, dx0, r_in=R1, ray_subsample=RS, n_workers=8,
            final_distance=TRAILING, focus_readout=_fr, final_leg='auto',
            **_ckw, **({'traced_kwargs': _tkw} if _tkw else {}))
    print(f"chain done {time.time()-t0:.0f}s")
    gi.assert_no_grid_degradation(_wrec, NFC, label="design-121 focus scan")

    dxo = 0.05e-6


    def metrics(E):
        E = np.asarray(E)
        n = E.shape[-1]
        I = np.abs(E) ** 2
        iy, ix = np.unravel_index(np.argmax(I), I.shape)
        xx = (np.arange(n) - ix) * dxo
        yy = (np.arange(n) - iy) * dxo
        rr = np.sqrt(xx[None, :] ** 2 + yy[:, None] ** 2)
        nb = 500
        ring = np.clip((rr / dxo).astype(int), 0, nb)
        s = np.bincount(ring.ravel(), weights=I.ravel(), minlength=nb + 1)
        cnt = np.bincount(ring.ravel(), minlength=nb + 1)
        prof = (s[:nb] / np.maximum(cnt[:nb], 1)) / I[iy, ix]
        rb = (np.arange(nb) + 0.5) * dxo
        idx = np.where(prof < 0.5)[0]
        fwhm = 2 * rb[idx[0]] if len(idx) else np.nan
        ee = {r: float(I[rr <= r * 1e-6].sum()) * dxo * dxo / P_in
              for r in (3, 6, 12)}
        off = ((ix - n // 2) * dxo, (iy - n // 2) * dxo)
        return fwhm, ee, float(I.max()), off


    E0 = np.asarray(res.field)
    fw, ee, pk, off = metrics(E0)
    print(f"AT-PLANE: FWHM={fw*1e6:.3f}um EE3={ee[3]*100:.1f}% EE6={ee[6]*100:.1f}% "
          f"EE12={ee[12]*100:.1f}% off=({off[0]*1e6:+.2f},{off[1]*1e6:+.2f})um")

    # BEST-FOCUS CRITERION (niche C6, 2026-07-30).  Historically this scan
    # selected on EE6 alone.  EE6 SATURATES near 99.7 % on a corrected spot, so it
    # discriminates on its 4th digit -- and once C6 removed the residual defocus it
    # picked dz = +10 um (EE6 99.7, EE3 85.3, FWHM 3.650) over dz = 0 (EE6 99.7,
    # EE3 90.2, FWHM 3.450, and the LARGEST peak): a plane 4.9 EE3 points and
    # 0.2 um worse, on an EE6 lead too small to print.  That reading cost a review
    # cycle, so both criteria are reported now.  ``pk`` (max intensity) is the
    # standard definition of best focus and is what the physics means here; the EE6
    # line is kept so the historical acceptance number stays directly comparable.
    best = (0.0, -1.0, None)          # EE6-selected (historical)
    best_pk = (0.0, -1.0, None)       # peak-intensity-selected (physical)
    for dz_um in range(-80, 81, 5):
        if dz_um == 0:
            Ez = E0
        else:
            Ez = angular_spectrum_propagate_mft(
                E0, dz_um * 1e-6, lam, dxo, dxo, NOUT)
        fw, ee, pk, off = metrics(Ez)
        if ee[6] > best[1]:
            best = (dz_um, ee[6], (fw, ee, pk, off))
        if pk > best_pk[1]:
            best_pk = (dz_um, pk, (fw, ee, pk, off))
        print(f"  dz={dz_um:+4d}um: FWHM={fw*1e6:6.3f} EE3={ee[3]*100:5.1f} "
              f"EE6={ee[6]*100:5.1f} EE12={ee[12]*100:5.1f} pk={pk:.3e}",
              flush=True)
    fwp, eep, pkp, offp = best_pk[2]
    print(f"BEST-FOCUS[peak] dz={best_pk[0]:+d}um: FWHM={fwp*1e6:.3f}um "
          f"EE3={eep[3]*100:.1f}% EE6={eep[6]*100:.1f}% EE12={eep[12]*100:.1f}% "
          f"pk={pkp:.3e}  <- physical best focus")
    fw, ee, pk, off = best[2]
    print(f"BEST-FOCUS dz={best[0]:+d}um: FWHM={fw*1e6:.3f}um EE3={ee[3]*100:.1f}% "
          f"EE6={ee[6]*100:.1f}% EE12={ee[12]*100:.1f}% "
          f"off=({off[0]*1e6:+.2f},{off[1]*1e6:+.2f})um")
    print("  targets: FWHM 3.223um EE3 91.0% EE6 100.0% (Zemax POP waist 2.737um "
          "radius -- of the paraxially-EQUIVALENT 4f, f1=60.916/f2=41.666, NOT of "
          "the real prescription (POP_CROSSCHECK_121_2026_07_31.md sec 2); "
          "ideal-field ceiling through this readout 3.45-3.55um / 90.3 / 99.8)")
    print("  shipping-default acceptance (CREF/AM/PIP unset, N=2048/NFC=8192/"
          "WF=4.0): AT-PLANE dz=0: 3.350um / 90.3 / 99.7 / 99.8, on-axis")
    print("  (re-baselined 2026-08-02, user-approved, with niche C9+C10: the exact "
          "sphere conversion and the r^6 residual term removed ~1 point/order of "
          "SIMULATOR error that was inflating the spot; 3.350um matches the "
          "exact-ray oracle to 0.02-0.06um.  Prior recorded lines: 3.450/90.2 "
          "at-plane post-C6, 3.450/88.8/99.6 at dz=+10um pre-C6.)")
    print("  (re-baselined 2026-08-01, user-approved: pre-C6 the chain focused "
          "10um PAST the MS plane -- at-plane read 3.750/87.4 and the recorded "
          "3.450/88.8/99.6 acceptance was taken at dz=+10um.  That offset was the "
          "simulator's own residual aberration; niche C6 removed it, so the spot "
          "is now sharpest AT the metasurface plane and is scored there.)")
