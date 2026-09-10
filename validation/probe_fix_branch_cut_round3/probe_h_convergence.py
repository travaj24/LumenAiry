"""ROUND 3: is the FD reference CONVERGING (a real derivative) or is it an
amplified band-level forward jump?  Central FD over a decade ladder in h."""
import probe_grad_mechanism as P
import jax, jax.numpy as jnp, numpy as np

CASES = (("w9  sum(R)", P._f_w9, 0.0), ("v514 sum(T)", P._f_v514, 0.0))
HS = (3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7)

for nm, fn in (("pre-round-1", P._decay_pre),
               ("round-1/2 conj(r)", P._decay_conj),
               ("cand A  -r", P._decay_neg),
               ("cand C  conj value / -r deriv", P._decay_sg)):
    P._patch(fn)
    print(f"\n=== {nm} ===")
    for label, f, th in CASES:
        ad = float(jax.grad(f)(jnp.asarray(th)))
        row = "  ".join(f"{P._central(f, th, h): .4e}" for h in HS)
        print(f"  {label:12s} AD={ad: .6e}")
        print(f"     h={'  '.join(f'{h:9.0e}' for h in HS)}")
        print(f"    FD= {row}")
