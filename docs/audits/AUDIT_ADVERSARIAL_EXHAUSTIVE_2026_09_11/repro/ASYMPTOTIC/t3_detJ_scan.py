"""T3: how much does |det J| (and hence the missing sqrt(detJ) amplitude
factor) vary over the alive output field, for progressively faster /
more-aberrated systems and full-box grids?"""
import sys, time
import numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import lumenairy as lm
from lumenairy.propagators.asymptotic import (
    fit_canonical_polynomials, propagate_modal_asymptotic,
    _solve_envelope_stationary_batch, _compute_M_b_batch,
)

def probe(R1, R2, d, obj, pupil_half, w_s, w_p, gridfrac, n=41, lam=1.31e-6):
    rx = lm.make_singlet(R1=R1, R2=R2, d=d, glass='N-BK7', aperture=40e-3)
    rx['object_distance'] = obj
    fit = fit_canonical_polynomials(rx, wavelength=lam,
                                    source_box_half=w_s*2, pupil_box_half=pupil_half,
                                    n_field=8, n_pupil=8, poly_order=6)
    L = fit.s2x_halfrange*gridfrac
    ax = np.linspace(-L, L, n) + fit.s2x_centre
    ay = np.linspace(-L, L, n) + fit.s2y_centre
    X, Y = np.meshgrid(ax, ay, indexing='xy')
    vx, vy, _ = _solve_envelope_stationary_batch(
        fit, X.ravel(), Y.ravel(), 0.0, 0.0, w_s=w_s, w_p=w_p,
        v_cx=fit.v2x_centre, v_cy=fit.v2y_centre)
    M, b, s1s, J, phis, G0, detJ = _compute_M_b_batch(
        fit, X.ravel(), Y.ravel(), vx, vy, 0.0, 0.0, w_s, w_p,
        fit.v2x_centre, fit.v2y_centre)
    E = propagate_modal_asymptotic(fit, source_point=(0.0,0.0), w_s=w_s, w_p=w_p,
                                   v2_centre=(fit.v2x_centre, fit.v2y_centre),
                                   s2_grid_x=X, s2_grid_y=Y)
    amp = np.abs(E).ravel()
    nz = amp > amp.max()*1e-6
    if nz.sum() < 4:
        return None
    dJ = detJ[nz]
    sq = np.sqrt(dJ)
    return dict(res=fit.res_phi_rms_waves, nalive=int(nz.sum()),
                ratio=float(dJ.max()/dJ.min()),
                sqrt_ratio=float(sq.max()/sq.min()),
                halfrange=fit.s2x_halfrange)

cases = [
    ("slow f/5 singlet, 30% box",  dict(R1=20e-3,R2=-20e-3,d=2e-3,obj=0.1,pupil_half=0.02,w_s=20e-6,w_p=0.02,gridfrac=0.3)),
    ("slow f/5 singlet, 95% box",  dict(R1=20e-3,R2=-20e-3,d=2e-3,obj=0.1,pupil_half=0.02,w_s=20e-6,w_p=0.02,gridfrac=0.95)),
    ("fast NA .15, 95% box",       dict(R1=20e-3,R2=-20e-3,d=2e-3,obj=0.1,pupil_half=0.15,w_s=20e-6,w_p=0.12,gridfrac=0.95)),
    ("fast NA .25, 95% box",       dict(R1=20e-3,R2=-20e-3,d=2e-3,obj=0.1,pupil_half=0.25,w_s=20e-6,w_p=0.2,gridfrac=0.95)),
    ("plano-convex NA .25",        dict(R1=20e-3,R2=np.inf,d=4e-3,obj=0.15,pupil_half=0.25,w_s=20e-6,w_p=0.2,gridfrac=0.95)),
    ("collimated-ish obj 1.0 NA.3",dict(R1=25e-3,R2=-25e-3,d=4e-3,obj=1.0,pupil_half=0.30,w_s=50e-6,w_p=0.25,gridfrac=0.95)),
]
for name, kw in cases:
    try:
        r = probe(**kw)
    except Exception as ex:
        print(f"{name:32s}  FAILED: {type(ex).__name__}: {ex}")
        continue
    if r is None:
        print(f"{name:32s}  no alive pixels"); continue
    print(f"{name:32s} res={r['res']:.2e}w alive={r['nalive']:5d} "
          f"detJ max/min={r['ratio']:.4f}  sqrt={r['sqrt_ratio']:.4f}")
