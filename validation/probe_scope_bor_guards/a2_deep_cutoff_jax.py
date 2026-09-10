"""CLASS A on BOR -- the DEEP cutoff mount, the JAX-twin parity, and the
build/thread signature.

WHY A DEEP CUTOFF.  The Cartesian band's BINDING population was not an
ordinary layer but a LAYER CUTOFF: for ``lam^2 = -s + i eta`` the principal
root's real part is ``eta / (2 sqrt(s))``, so at fixed backward error the
discriminating ratio grows WITHOUT LIMIT as ``s -> 0`` and no constant band
can be proved safe.  The cylindrical peer of ``s -> 0`` is a radial order at
its own cutoff: ``q^2 = k0^2 eps - gamma_j^2 -> 0``, whence

    rho = |Im q| / |Re q| ~ |Im q^2| / (2 (Re q)^2)   ~   1 / qn^2 .

The PEC-walled cylindrical spectrum is discrete, so an ordinary ``k0`` sweep
cannot reach the degenerate point; here the cutoff of one NAMED order is
solved for and approached geometrically.  With ``k0 = gamma_j / (n
sqrt(1 - delta))`` the order sits at exactly ``qn = n sqrt(delta)``, so a
``delta`` ladder 1e-04 .. 1e-26 spans ``qn`` = 1.4e-02 .. 1.4e-13.

Reported per rung: ``rho`` (the NOISE side of the 1e-9 band) and ``relflux``
(the flux normalizer's fallback fires at 1e-10 -- below it the orientation is
the sign of rounding).  Also here: the JAX twins carry their OWN copies of the
orientation rule, so their R / T must match the NumPy path mode for mode; and
a compact per-fixture signature (the flux-sign string plus R / T / closure) for
the cross-build and thread-ladder comparison.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import closure, dump, mode_table, pin_tree  # noqa: E402

print("TREE", pin_tree())
RBIG = 24.0
NFD = 120
LAM = 3.0


def fd_modes(m, k0, eps, N=NFD):
    from lumenairy.elements.bor.zcascade import layer_modes
    return layer_modes(m, RBIG, N,
                       lambda r, e=eps: np.full_like(r, e, dtype=complex),
                       float(k0), staggered=True)


def sem_modes(m, k0, eps, degree=8):
    from lumenairy.elements.bor.sem_radial import SemRadialMesh, sem_layer_modes
    n = abs(np.sqrt(complex(eps)).real)
    lam_loc = 2 * np.pi / (k0 * max(n, 1e-3))
    max_el = degree * lam_loc / 8.0             # the shipped DPW = 8 cap
    ne = max(2, int(np.ceil(RBIG / max_el)))
    b = np.linspace(0.0, RBIG, ne + 1)
    mesh = SemRadialMesh(b, [(complex(eps),) * 3] * ne, degree)
    return sem_layer_modes(mesh, m, float(k0))


def gammas(m, eps, *, sem=False):
    k0 = 2.0
    L = sem_modes(m, k0, eps) if sem else fd_modes(m, k0, eps)
    q = np.asarray(L["q"])
    g = np.sqrt(k0 ** 2 * complex(eps) - q ** 2)
    g = np.real(g[np.abs(g.imag) < 1e-9 * np.maximum(np.abs(g.real), 1e-300)])
    return np.sort(g[g > 1e-6])


def ladder(m, eps, *, sem, gamma, deltas):
    n = abs(np.sqrt(complex(eps)).real)
    rows = []
    for dl in deltas:
        k0 = gamma / (n * np.sqrt(1.0 - dl))
        L = sem_modes(m, k0, eps) if sem else fd_modes(m, k0, eps)
        q, flux, rel, rho = mode_table(L, sem=sem)
        qn = q / k0
        prop = rho < 1e-9
        if prop.any():
            j = int(np.argmin(np.abs(qn.real)[prop]))
            rows.append(dict(
                delta=float(dl), k0=float(k0), sem=bool(sem), m=m,
                qn_target=float(n * np.sqrt(dl)),
                qn_found=float(np.abs(qn.real)[prop][j]),
                rho=float(rho[prop][j]), relflux=float(rel[prop][j]),
                n_prop=int(prop.sum()),
                kept_by_solve_gate=bool(abs(qn[prop][j].imag) < 5e-5
                                        and qn[prop][j].real > 1e-6),
                rho_max_all_prop=float(rho[prop].max()),
                relflux_min_all_prop=float(rel[prop].min())))
        else:
            jj = int(np.argmin(np.abs(qn.real)))
            rows.append(dict(delta=float(dl), k0=float(k0), sem=bool(sem),
                             m=m, qn_target=float(n * np.sqrt(dl)),
                             qn_found=float(np.abs(qn.real)[jj]),
                             rho=float(rho[jj]), relflux=float(rel[jj]),
                             n_prop=0, kept_by_solve_gate=False,
                             rho_max_all_prop=None,
                             relflux_min_all_prop=None))
    return rows


def jax_parity():
    """The JAX twins carry their own copies of the orientation rule."""
    try:
        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
    except Exception as exc:                       # noqa: BLE001
        return dict(available=False, why=str(exc)[:200])
    from lumenairy import BORStack
    out = []
    for basis, m in (("fd", 0), ("fd", 1), ("sem", 0), ("sem", 1)):
        try:
            def run(epsval, basis=basis, m=m):
                s = BORStack(RBIG, m, n_substrate=1.41, n_superstrate=1.41,
                             N=NFD, basis=basis, degree=8)
                s.add_layer(0.4, eps=epsval)
                if basis == "sem":
                    s.add_layer(0.5, segments=[(3.0, 6.0), (RBIG, 1.41 ** 2)])
                else:
                    s.add_layer(0.5, rings=(LAM, 0.5, 2.45, 1.41))
                s.set_source(k0=2.0)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return s.solve()
            rn = run(complex(1.41 ** 2))
            rj = run(jnp.asarray(1.41 ** 2 + 0j))
            Rn, Tn = float(np.sum(rn["R"])), float(np.sum(rn["T"]))
            Rj = float(np.sum(np.asarray(rj["R"])))
            Tj = float(np.sum(np.asarray(rj["T"])))
            out.append(dict(basis=basis, m=m, R_np=Rn, T_np=Tn, R_jax=Rj,
                            T_jax=Tj, dR=abs(Rn - Rj), dT=abs(Tn - Tj)))
        except Exception as exc:                   # noqa: BLE001
            out.append(dict(basis=basis, m=m,
                            error=f"{type(exc).__name__}: {exc}"[:200]))
    return dict(available=True, rows=out)


def signature():
    from lumenairy import BORStack
    out = []
    for n, nm in ((1.41, "141"), (2.00, "200"), (1.50 + 0.05j, "l150")):
        for m in (0, 1, 2):
            for basis in ("fd", "sem"):
                eps = complex(n) ** 2
                for kind in ("coincide", "spacers"):
                    s = BORStack(RBIG, m, n_substrate=n, n_superstrate=n,
                                 N=NFD, basis=basis, degree=8)
                    s.add_layer(0.4, eps=eps)
                    if kind == "spacers":
                        s.add_layer(0.5, rings=(LAM, 0.5, 2.45, 1.41))
                        s.add_layer(0.4, eps=eps)
                    s.set_source(k0=2.0)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        res = s.solve()
                    d = s._last
                    sem = basis == "sem"
                    sig = []
                    for tag, L in ([("sup", d["sup"]), ("sub", d["sub"])]
                                   + [(f"mid{i}", LL) for i, (_t, LL)
                                      in enumerate(d["mids"])]):
                        q, flux, rel, rho = mode_table(L, sem=sem)
                        sig.append(f"{tag}:{int((rho < 1e-9).sum())}:"
                                   + "".join("+" if v >= 0 else "-"
                                             for v in flux))
                    out.append(dict(id=f"{kind}_{nm}_m{m}_{basis}",
                                    sign="|".join(sig),
                                    R=float(np.sum(res["R"])),
                                    T=float(np.sum(res["T"])),
                                    closure=closure(res),
                                    n_orders=int(np.size(res["R"]))))
    return out


def main():
    tag = os.environ.get("PROBE_TAG", "win")
    thr = os.environ.get("OPENBLAS_NUM_THREADS", "?")
    payload = dict(threads=thr, ladders=[], sig=signature())
    if os.environ.get("PROBE_FULL", "1") == "1":
        deltas = [10.0 ** (-e) for e in range(4, 27, 2)]
        for sem in (False, True):
            for m in (0, 1):
                eps = 1.41 ** 2
                g = gammas(m, eps, sem=sem)
                if g.size < 3:
                    continue
                gamma = float(g[2])
                rows = ladder(m, eps, sem=sem, gamma=gamma, deltas=deltas)
                payload["ladders"].append(dict(sem=sem, m=m, gamma=gamma,
                                               rows=rows))
                print(f"\n== {'SEM' if sem else 'FD '} m={m} "
                      f"gamma={gamma:.6f} ==")
                print("  delta      qn_target    qn_found     rho          "
                      "relflux      n_prop  kept")
                for r in rows:
                    print(f"  {r['delta']:.0e}   {r['qn_target']:.4e}   "
                          f"{r['qn_found']:.4e}   {r['rho']:.4e}   "
                          f"{r['relflux']:.4e}   {r['n_prop']:4d}   "
                          f"{r['kept_by_solve_gate']}")
        payload["jax"] = jax_parity()
        print("\n== JAX twin parity ==")
        for r in (payload["jax"].get("rows") or []):
            print("  ", r)
        allr = [r for L in payload["ladders"] for r in L["rows"]]
        print("\n== HEADLINE ==")
        print("  rungs:", len(allr))
        print("  rungs whose NOISE side reaches the 1e-9 bar:",
              sum(1 for r in allr if r["rho"] >= 1e-9))
        print(f"  worst noise-side rho: {max(r['rho'] for r in allr):.4e} "
              f"({np.log10(1e-9 / max(r['rho'] for r in allr)):.2f} decades "
              f"below the bar)")
        print(f"  min relflux: {min(r['relflux'] for r in allr):.4e} "
              f"(fallback fires at 1e-10)")
    dump(f"a2_deepcutoff_{tag}_t{thr}.json", payload)
    print("  signature fixtures:", len(payload["sig"]), " closure max:",
          max(x["closure"] for x in payload["sig"]
              if x["closure"] == x["closure"]))


if __name__ == "__main__":
    main()
