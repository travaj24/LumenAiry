import numpy as np, warnings, time
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
inc = np.real(np.asarray(res["inc"]))
out = np.real(np.asarray(res["out"]))
ii = np.where(inc > 0.5)[0]
oo = np.where(out > 0.5)[0]
A11 = S11[np.ix_(oo if False else ii, ii)]
# S11 maps incident (sup) -> reflected (sup); rows are sup channels
A11 = S11[np.ix_(ii, ii)]
A21 = S21[np.ix_(oo, ii)]
e = np.asarray(res["energy"], float)
print(f"n_inc={len(ii)} n_out={len(oo)}  per-channel max|E-1|={np.max(np.abs(e-1)):.3e}",
      flush=True)
# per-channel closure recomputed from the restricted blocks
col = (np.sum(np.abs(A11)**2, axis=0) + np.sum(np.abs(A21)**2, axis=0))
print(f"restricted per-column closure: max|.-1| = {np.max(np.abs(col-1)):.3e}",
      flush=True)
# SUPERPOSITION energy: excite with random unit-norm combinations
rng = np.random.default_rng(0)
worst = 0.0
for t in range(200):
    c = rng.standard_normal(len(ii)) + 1j*rng.standard_normal(len(ii))
    c /= np.linalg.norm(c)
    r = A11 @ c
    tt = A21 @ c
    tot = np.sum(np.abs(r)**2) + np.sum(np.abs(tt)**2)
    worst = max(worst, abs(tot-1.0))
print(f"SUPERPOSITION closure over 200 random inputs: worst |R+T-1| = "
      f"{worst:.6e}", flush=True)
# two-channel superpositions, to localise
w2 = 0.0
pair = None
for a in range(min(len(ii), 25)):
    for b in range(a+1, min(len(ii), 25)):
        c = np.zeros(len(ii), complex); c[a] = 1/np.sqrt(2); c[b] = 1/np.sqrt(2)
        tot = np.sum(np.abs(A11@c)**2) + np.sum(np.abs(A21@c)**2)
        if abs(tot-1) > w2:
            w2, pair = abs(tot-1), (a, b)
print(f"worst 2-channel pair {pair}: |R+T-1| = {w2:.6e}", flush=True)
M = np.vstack([A11, A21])
G = M.conj().T @ M
print(f"Gram |G - I| = {np.max(np.abs(G-np.eye(G.shape[0]))):.3e}  "
      f"(diag dev {np.max(np.abs(np.diag(G)-1)):.3e})", flush=True)
