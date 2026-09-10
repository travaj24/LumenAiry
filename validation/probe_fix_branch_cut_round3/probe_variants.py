"""ROUND 3: run every candidate flip through the gradient gates."""
import probe_grad_mechanism as P   # noqa: E402  (same directory)

if __name__ == "__main__":
    for nm, fn in (("pre-round-1 (exact ==0 pin)", P._decay_pre),
                   ("round-1/2 SHIPPED conj(r)", P._decay_conj),
                   ("cand A  -r  (holomorphic)", P._decay_neg),
                   ("cand B  -r + 1j*imag zeroing", P._decay_imagzero),
                   ("cand C  conj value / -r derivative", P._decay_sg)):
        print(f"\n=== {nm} ===")
        for label, ad, fd, rel in P.report(nm, fn):
            print(f"  {label:24s} AD={ad: .6e}  FD={fd: .6e}  rel={rel:.3e}")
