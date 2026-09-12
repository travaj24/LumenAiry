import numpy as np, time
t0 = time.perf_counter()
from lumenairy.elements.bor import BORStack
print("import %.1f s" % (time.perf_counter() - t0), flush=True)

S = 1e6
k0r = 2.0*np.pi/(1.0e-6*S)
s = BORStack(Rbig=12e-6*S, m=1, N=96, n_superstrate=1.41+0j,
             n_substrate=1.41+0j)
s.add_layer(0.5e-6*S, rings=(3.0e-6*S, 0.5, 2.45+0j, 1.41+0j))
s.set_source(k0=k0r)
res = s.solve()
S11, S12, S21, S22 = [np.asarray(x) for x in res["S"]]
inc = np.asarray(res["inc"], dtype=int)      # INDEX array, not a mask
out = np.asarray(res["out"], dtype=int)
A11 = S11[np.ix_(inc, inc)]
A21 = S21[np.ix_(out, inc)]
A12 = S12[np.ix_(inc, out)]
A22 = S22[np.ix_(out, out)]
e = np.asarray(res["energy"], float)
print(f"n_inc={len(inc)} n_out={len(out)}  library max|E-1|={np.max(np.abs(e-1)):.3e}",
      flush=True)
M = np.vstack([A11, A21])
col = np.sum(np.abs(M)**2, axis=0)
print(f"restricted per-column closure: max|.-1| = {np.max(np.abs(col-1)):.3e}",
      flush=True)
G = M.conj().T @ M
off = G - np.diag(np.diag(G))
print(f"Gram: max|offdiag| = {np.max(np.abs(off)):.3e}   "
      f"max|diag-1| = {np.max(np.abs(np.diag(G)-1)):.3e}", flush=True)
rng = np.random.default_rng(0)
worst = 0.0
for _ in range(300):
    c = rng.standard_normal(len(inc)) + 1j*rng.standard_normal(len(inc))
    c /= np.linalg.norm(c)
    tot = np.sum(np.abs(A11 @ c)**2) + np.sum(np.abs(A21 @ c)**2)
    worst = max(worst, abs(tot-1.0))
print(f"SUPERPOSITION closure over 300 random inputs: worst |R+T-1| = "
      f"{worst:.6e}", flush=True)
U = np.block([[A11, A12], [A21, A22]])
print(f"full propagating S: |U^H U - I| = "
      f"{np.max(np.abs(U.conj().T@U - np.eye(U.shape[0]))):.3e}", flush=True)
# gauge-fixed reciprocity via the library's own pinned-phase accessor
try:
    ar = s.per_mode_amplitudes(port="reflection")["amplitude"]
    at = s.per_mode_amplitudes(port="transmission")["amplitude"]
    print(f"pinned-gauge |S11 - S11^T|/max = "
          f"{np.max(np.abs(ar-ar.T))/max(np.max(np.abs(ar)),1e-30):.3e}",
          flush=True)
    print(f"pinned-gauge |S21 - S21^T|/max = "
          f"{np.max(np.abs(at-at.T))/max(np.max(np.abs(at)),1e-30):.3e}",
          flush=True)
except Exception as ex:
    print("  per_mode_amplitudes:", type(ex).__name__, str(ex)[:150], flush=True)
