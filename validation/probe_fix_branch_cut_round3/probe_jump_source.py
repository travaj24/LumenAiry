"""ROUND 3: sweep theta finely; at each theta record the FLIP MASK, the flipped
modes' Re(r), and f -- so the forward jump can be attributed."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from lumenairy.elements.rcwa import _core as C

MODE = os.environ.get("BC3_MODE", "conj")
SEEN = []
_ORIG = C._sqrt_decay


def _decay(x, xp=None, band=C._CUT_BAND_REL):
    r = np.sqrt(np.asarray(x).astype(C._C))
    scale = max(float(np.max(np.abs(r))), 1.0) if r.size else 1.0
    flip = (np.abs(r.real) <= band * scale) & (r.imag < 0)
    if r.size > 4:
        SEEN.append((tuple(np.where(flip)[0]),
                     tuple(np.round(r.real[flip] / (band * scale), 6)),
                     float(np.max(r.real[flip])) if flip.any() else 0.0))
    if MODE == "conj":
        return np.where(flip, np.conj(r), r)
    return r * np.where(flip, -1.0, 1.0)


C._sqrt_decay = _decay
import lumenairy.elements.pmm.twod as T
T._sqrt_decay = _decay
from lumenairy.elements.pmm import pmm_efficiency_2d

P, WL, DEP = 0.6e-6, 0.55e-6, 0.25e-6
XB = (0.2 * P, 0.6 * P)
TH = np.linspace(-4e-6, 4e-6, 41)
vals, masks, amax = [], [], []
for t in TH:
    SEEN.clear()
    o, R, Tt = pmm_efficiency_2d(P, P, 6.0 + 0j, 1.0, XB, XB, 1.5, 1.0, DEP,
                                 WL, theta=float(t), degree=5, n_orders=2,
                                 polarization="te")
    vals.append(float(np.sum(Tt)))
    masks.append(SEEN[-1][0] if SEEN else ())
    amax.append(SEEN[-1][2] if SEEN else 0.0)
vals = np.array(vals); amax = np.array(amax)
d = np.abs(np.diff(vals))
smooth = 6.2788e-02 * (TH[1] - TH[0])
print(f"MODE={MODE}  worst step {d.max():.4e}  vs smooth {smooth:.4e}  "
      f"ratio {d.max()/smooth:.3f}")
print(f"flip-mask constant across sweep? "
      f"{len(set(masks)) == 1}   distinct masks: {len(set(masks))}")
for m in sorted(set(masks)):
    print(f"   mask {m}: {masks.count(m)} thetas")
print(f"max Re(r) of a flipped mode over the sweep: "
      f"{amax.min():.4e} .. {amax.max():.4e}")
k = int(np.argmax(d))
print(f"worst step at theta {TH[k]:.3e} -> {TH[k+1]:.3e}: "
      f"masks {masks[k]} -> {masks[k+1]};  a {amax[k]:.4e} -> {amax[k+1]:.4e}")
