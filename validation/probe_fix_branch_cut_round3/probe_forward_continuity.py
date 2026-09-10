"""ROUND 3: the forward answer f(theta) on a FINE theta grid, under each flip
rule.  A step here is what the failing gates' h=1e-6 central difference was
dividing by 2e-6 and calling a derivative."""
import probe_grad_mechanism as P
import jax.numpy as jnp, numpy as np

TH = np.linspace(-4e-6, 4e-6, 41)

for nm, fn in (("pre-round-1", P._decay_pre),
               ("round-1/2 conj(r)", P._decay_conj),
               ("cand A  -r", P._decay_neg)):
    P._patch(fn)
    v = np.array([float(P._f_v514(jnp.asarray(t))) for t in TH])
    lin = np.polyval(np.polyfit(TH, v, 1), TH)
    resid = v - lin
    d = np.diff(v)
    print(f"\n=== {nm} ===")
    print(f"  f range           {v.min():.12f} .. {v.max():.12f}")
    print(f"  worst |f - linear fit|          {np.max(np.abs(resid)):.4e}")
    print(f"  worst |f(k+1)-f(k)| (dtheta=2e-7) {np.max(np.abs(d)):.4e}")
    print(f"  smooth expectation |f'|*dtheta    "
          f"{abs(6.2788e-02) * (TH[1]-TH[0]):.4e}")
    print(f"  JUMP RATIO (worst step / smooth)  "
          f"{np.max(np.abs(d)) / (abs(6.2788e-02) * (TH[1]-TH[0])):.4e}")
