"""TASK E / claim 3 -- the EME branch cut.  The FAST battery.

Runs every strip-level, mode-match, diffraction and vector fixture on ONE tree
(``--build pre|post``) and one arm, hashes the answers (SHA-256 of the exact
IEEE-754 bytes) and records the flipped-mode census, the band's two-sided
population and the ``|exp(i ky h)| <= 1`` cascade guarantee.

Every quantity the verdict rests on is computed HERE from the raw spectrum, so
nothing is read out of the build's own comments.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _vh  # noqa: E402
import ve_fix  # noqa: E402


# --------------------------------------------------------------------------- #
#  THE TWO RULES, implemented HERE (never imported from the tree under test)   #
# --------------------------------------------------------------------------- #
def rule_pin(z):
    """The pre-5.45.1 exact-zero pin."""
    return np.where(z.imag < 0.0, -z, z)


def rule_band(z, band=1e-9):
    """The 5.45.1 rule, re-derived from its own description: off the cut
    negate, on the cut conjugate, band relative to the spectrum's top and
    floored at 1.0."""
    tol = band * max(1.0, float(np.max(np.abs(z))))
    neg = z.imag < 0.0
    on_cut = np.abs(z.imag) <= tol
    return np.where(neg, np.where(on_cut, np.conj(z), -z), z)


def classify(z0):
    """Per-mode class from the PRINCIPAL root, measured not assumed.

    ``prop``  : a near-real root (|Im| < 1e-6 |Re|, Re > 0) -- the population
                whose Im is backward error.
    ``evan``  : a near-imaginary root (|Re| < 1e-6 |Im|) -- lossless evanescent.
    ``lossy`` : everything else, i.e. a genuinely complex root.
    """
    a, b = np.abs(z0.real), np.abs(z0.imag)
    prop = (z0.real > 0) & (b < 1e-6 * a)
    evan = a < 1e-6 * b
    lossy = ~(prop | evan)
    return prop, evan, lossy


def census(z0, out):
    """Which of the three possible outputs each mode got, exactly."""
    same = (out == z0)
    neg = (out == -z0) & ~same
    con = (out == np.conj(z0)) & ~same & ~neg
    other = ~(same | neg | con)
    return same, neg, con, other


def strip_record(eme_2d, name, spec):
    eps = ve_fix.build_eps(spec)
    Nx, Lx, k0, kx0, qz2 = (spec["Nx"], spec["Lx"], spec["k0"], spec["kx0"],
                            spec["qz2"])
    lam_raw, Phi = eme_2d.strip_x_modes(eps, Lx, Nx, k0, kx0)
    lam_raw = np.asarray(lam_raw, dtype=complex)
    order = np.lexsort((lam_raw.imag, lam_raw.real))
    lam = lam_raw[order]
    z0 = np.sqrt(lam - qz2 + 0j)
    out = np.asarray(eme_2d._ky_forward(lam, qz2))
    out_raw = np.asarray(eme_2d._ky_forward(lam_raw, qz2))

    prop, evan, lossy = classify(z0)
    same, neg, con, other = census(z0, out)
    scale = max(1.0, float(np.max(np.abs(z0))))
    ratio = np.abs(z0.imag) / scale
    # THE ARM-INVARIANT OBSERVABLE.  The raw spectrum's last bits move with the
    # LAPACK kernel on BOTH builds (that is the background, not the signal), so
    # the decision is recorded as a per-mode CODE relative to the principal
    # root -- 0 same, 1 negated, 2 conjugated, 3 neither -- which is what the
    # branch rule actually decides, and is what must not move with the arm.
    code = np.zeros(z0.size, dtype=np.int8)
    code[neg] = 1
    code[con] = 2
    code[other] = 3
    n_oncut = int(np.sum(np.abs(z0.imag) <= 1e-9 * scale))
    h = 0.5 * Lx
    growth = float(np.max(np.abs(np.exp(1j * out * h))))
    growth_raw = float(np.max(np.abs(np.exp(1j * out_raw * h))))

    def mx(mask, arr):
        return float(np.max(arr[mask])) if np.any(mask) else None

    def mn(mask, arr):
        return float(np.min(arr[mask])) if np.any(mask) else None

    return dict(
        name=name, spec={k: (float(v) if isinstance(v, (int, float)) else v)
                         for k, v in spec.items()},
        n=int(lam.size),
        hash_lam=_vh.hash_arrays(lam), hash_ky=_vh.hash_arrays(out),
        hash_ky_unsorted=_vh.hash_arrays(out_raw),
        hash_z0=_vh.hash_arrays(z0),
        hash_code=_vh.hash_arrays(code), n_oncut=n_oncut,
        n_prop=int(prop.sum()), n_evan=int(evan.sum()), n_lossy=int(lossy.sum()),
        n_same=int(same.sum()), n_neg=int(neg.sum()), n_conj=int(con.sum()),
        n_other=int(other.sum()),
        # THE CENSUS: a mode whose returned root is not the principal one
        flip_prop=int((neg & prop).sum()), flip_evan=int((neg & evan).sum()),
        flip_lossy=int((neg & lossy).sum()),
        conj_prop=int((con & prop).sum()), conj_evan=int((con & evan).sum()),
        conj_lossy=int((con & lossy).sum()),
        # the worst root MOVE, relative to the root's own size
        worst_dky=float(np.max(np.abs(out - z0))),
        worst_dky_rel=float(np.max(np.abs(out - z0)
                                   / np.maximum(np.abs(z0), 1e-300))),
        # THE BAND's two sides, measured
        scale=scale, band=1e-9 * scale,
        oncut_pop_max=mx(prop, ratio),          # backward error it must absorb
        lossy_pop_min=mn(lossy, ratio),         # physics it must not absorb
        evan_pop_min=mn(evan, ratio),
        min_absz=float(np.min(np.abs(z0))),
        max_absz=float(np.max(np.abs(z0))),
        # the cascade guarantee
        growth=growth, growth_unsorted=growth_raw,
        min_imag_out=float(np.min(out.imag)),
        # cross-check against the rule re-implemented HERE
        matches_pin=bool(np.array_equal(out, rule_pin(z0))),
        matches_band=bool(np.array_equal(out, rule_band(z0))),
    )


def fwd_ky(eme_2d, profile, Nx, Lx, k0, kx0, imag, qz2):
    """The forward ky SET of one strip, in the order of the lam spectrum sorted
    by (Re, Im) -- the construction the build's own gate uses, so the two
    eigensolver routings (eigh for a real eps, scipy eig for any complex one)
    are compared elementwise."""
    eps = ve_fix.build_eps(dict(profile=profile, Nx=Nx, imag=imag))
    lam = np.asarray(eme_2d.strip_x_modes(eps, Lx, Nx, k0, kx0)[0],
                     dtype=complex)
    lam = lam[np.lexsort((lam.imag, lam.real))]
    return np.asarray(eme_2d._ky_forward(lam, qz2)), lam


def infinitesimal_pairs(eme_2d, out, arrays):
    """CLAIM (b), re-measured on MY fixtures: adding an INFINITESIMAL Im(eps)
    is a physical no-op, so the forward ky set must not move.  Reports how many
    of the N modes came back on a DIFFERENT root and the worst |d ky|."""
    rows = []
    base = [("split", 96, 1.0, 20 * np.pi, 0.0, 0.0),
            ("split", 128, 1.0, 20 * np.pi, 0.0, 0.0),
            ("split", 128, 1.0, 40 * np.pi, 0.0, 0.0),
            ("split", 48, 1.0, 20 * np.pi, 0.0, 0.0),
            ("split", 96, 1.0, 20 * np.pi, 0.0, 26055.8),
            ("centre", 96, 1.0, 20 * np.pi, 0.37, 0.0),
            ("three", 96, 1.0, 20 * np.pi, 0.0, 0.0),
            ("split", 96, 1e-3, 2e4 * np.pi, 0.0, 0.0),
            ("split", 96, 1e3, 2e-2 * np.pi, 0.0, 0.0)]
    for prof, Nx, Lx, k0, kx0, qz2 in base:
        ref, lam0 = fwd_ky(eme_2d, prof, Nx, Lx, k0, kx0, 0.0, qz2)
        tag = "%s_Nx%d_Lx%g_k%g_kx%g_qz%g" % (prof, Nx, Lx, k0, kx0, qz2)
        arrays["ref_" + tag] = ref
        for imag in (1e-30, 1e-20, 1e-12, 1e-6):
            got, lam1 = fwd_ky(eme_2d, prof, Nx, Lx, k0, kx0, imag, qz2)
            arrays["im%g_%s" % (imag, tag)] = got
            d = np.abs(got - ref)
            scale = np.maximum(np.abs(ref), 1.0)
            ndiff = int(np.sum(d > 1e-6 * scale))
            rows.append(dict(
                fixture=tag, imag=imag, n=int(ref.size), n_diff=ndiff,
                worst_dky=float(np.max(d)),
                worst_dky_rel=float(np.max(d / scale)),
                worst_dlam=float(np.max(np.abs(lam1 - lam0))),
                hash_ref=_vh.hash_arrays(ref), hash_got=_vh.hash_arrays(got)))
    out["infinitesimal"] = rows


# --------------------------------------------------------------------------- #
#  mode_match / diffraction / vector fixtures                                  #
# --------------------------------------------------------------------------- #
def _uniform_eps(Nx, Ny, n):
    return np.full((Nx, Ny), complex(n) ** 2)


def diffraction_records(ed, out):
    rows = []
    import warnings
    for name, n, depth, lossy in (
            ("fd_uniform_n15_d02", 1.5, 0.2, False),
            ("fd_uniform_n15_d20", 1.5, 2.0, False),
            ("fd_uniform_n15p02j_d40", complex(1.5, 0.2), 4.0, True),
            ("fd_uniform_n15p1e30j_d40", complex(1.5, 1e-30), 4.0, True),
            ("fd_uniform_n15p1e12j_d40", complex(1.5, 1e-12), 4.0, True),
            ("fd_uniform_n15m1e6j_d40", complex(1.5, -1e-6), 4.0, True),
    ):
        Nx = Ny = 8
        eps = _uniform_eps(Nx, Ny, n)
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = ed.diffraction_fd(eps, 1.0, 1.0, Nx, Ny, 8.0, 1.0, 1.0,
                                    depth, 1, 1, kx0=0.0, ky0=0.0)
        rows.append(dict(name=name, secs=round(time.time() - t0, 2),
                         hash_r=_vh.hash_arrays(res["r"]),
                         hash_t=_vh.hash_arrays(res["t"]),
                         hash_RT=_vh.hash_arrays(res["R"], res["T"]),
                         energy=float(res["energy"]),
                         R00=float(res["R"][res["orders"].index((0, 0))]),
                         T00=float(res["T"][res["orders"].index((0, 0))])))
    out["diffraction_fd"] = rows


def mode_match_records(ed, out):
    """mode_match driven DIRECTLY with a crafted qz2 spectrum, so the branch
    decision is exercised at values I choose rather than ones an eigensolve
    happens to produce."""
    rows = []
    Nx = Ny = 8
    orders = ed.plane_wave_orders(1, 1)
    K = len(orders)
    U, kx, ky = ed.pw_matrix(orders, 0.0, 0.0, 1.0, 1.0, Nx, Ny)
    Psi = U.copy()                                     # exact plane-wave basis
    qz2_base = 2.25 * 8.0 ** 2 - kx ** 2 - ky ** 2     # the analytic slab
    for name, d in (("exactreal", 0.0), ("im_m1e30", -1e-30),
                    ("im_m1e20", -1e-20), ("im_m1e12", -1e-12),
                    ("im_m1e6", -1e-6), ("im_m1e2", -1e-2),
                    ("im_p1e2", 1e-2)):
        qz2 = qz2_base.astype(complex) + 1j * d
        for depth in (0.2, 4.0):
            t0 = time.time()
            res = ed.mode_match(qz2, Psi, orders, kx0=0.0, ky0=0.0, k0=8.0,
                                eps_sup=1.0, eps_sub=1.0, depth=depth,
                                Lx=1.0, Ly=1.0, Nx=Nx, Ny=Ny)
            z0 = np.sqrt(np.asarray(qz2, complex) + 0j)
            rows.append(dict(name="mm_%s_d%g" % (name, depth),
                             secs=round(time.time() - t0, 2),
                             hash_r=_vh.hash_arrays(res["r"]),
                             hash_t=_vh.hash_arrays(res["t"]),
                             energy=float(res["energy"]),
                             T00=float(res["T"][orders.index((0, 0))]),
                             R00=float(res["R"][orders.index((0, 0))]),
                             min_im_z0=float(np.min(z0.imag)),
                             scale=float(max(1.0, np.max(np.abs(z0)))),
                             n_pin_flip=int(np.sum(z0.imag < 0.0))))
    out["mode_match"] = rows


def vector_records(ev, out):
    rows = []
    for name, imag, Nx, k0 in (("vec_real_Nx24", 0.0, 24, 8.0),
                               ("vec_1e30_Nx24", 1e-30, 24, 8.0),
                               ("vec_1e12_Nx24", 1e-12, 24, 8.0),
                               ("vec_1e3_Nx24", 1e-3, 24, 8.0),
                               ("vec_real_Nx48", 0.0, 48, 20.0 * np.pi),
                               ("vec_1e30_Nx48", 1e-30, 48, 20.0 * np.pi),
                               ("vec_gain_Nx24", -1e-6, 24, 8.0)):
        eps = ve_fix.eps_split(Nx, lo=2.25, hi=6.25, imag=imag)
        t0 = time.time()
        ky, W, V = ev.strip_vector_modes(eps, 1.0, Nx, k0, kx0=0.0, qz2=0.0)
        ky = np.asarray(ky)
        rows.append(dict(name=name, secs=round(time.time() - t0, 2),
                         n_fwd=int(ky.size),
                         hash_ky=_vh.hash_arrays(np.sort_complex(ky)),
                         hash_ky_raw=_vh.hash_arrays(ky),
                         hash_W=_vh.hash_arrays(W),
                         min_imag=float(np.min(ky.imag)),
                         max_absky=float(np.max(np.abs(ky))),
                         tol=1e-9 * max(1.0, float(np.max(np.abs(ky)))),
                         growth=float(np.max(np.abs(np.exp(1j * ky * 0.5))))))
    out["vector"] = rows


def split_forward_records(ev, out):
    """``_strip_split_forward`` on SYNTHETIC spectra -- the claim is that
    ``cut_band`` computes the identical quantity the inline expression did, so
    the index set must be identical on every one of these, INCLUDING the
    degenerate ones (empty, all-tiny, one huge)."""
    rows = []
    cases = {
        "plain": np.array([1 + 0j, -1 + 0j, 2j, -2j, 0.5 - 1e-12j]),
        "tiny": np.array([1e-12 + 0j, -1e-12 + 0j, 1e-12j, -1e-12j]),
        "huge": np.array([1e8 + 0j, -1e8 + 0j, 1.0 + 1e-3j, 1.0 - 1e-3j]),
        "oncut": np.array([1.0 + 1e-10j, 1.0 - 1e-10j, -1.0 + 1e-10j]),
        "single": np.array([3.0 + 0j]),
        "empty": np.array([], dtype=complex),
    }
    for name, ky in cases.items():
        try:
            idx = ev._strip_split_forward(ky)
            rows.append(dict(name=name, idx=[int(i) for i in idx],
                             hash_idx=_vh.hash_arrays(np.asarray(idx)),
                             err=None))
        except Exception as exc:                       # noqa: BLE001
            rows.append(dict(name=name, idx=None, hash_idx=None,
                             err="%s: %s" % (type(exc).__name__, exc)))
    out["split_forward"] = rows


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True, choices=["pre", "post"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--only", default="all")
    a = ap.parse_args()

    import lumenairy
    print("lumenairy.__file__ =", lumenairy.__file__)
    _vh.require_tree(a.build)
    from lumenairy.elements.eme import eme_2d, eme_2d_vector, eme_diffraction

    out = dict(task="E", claim=3, build=a.build, tag=a.tag, arm=_vh.arm(),
               lumenairy_file=lumenairy.__file__,
               eme_2d_file=eme_2d.__file__)
    t0 = time.time()
    arrays = {}
    if a.only in ("all", "strip"):
        rows = []
        for name, spec in ve_fix.strip_fixtures():
            rows.append(strip_record(eme_2d, name, spec))
            eps = ve_fix.build_eps(spec)
            lam = np.asarray(eme_2d.strip_x_modes(
                eps, spec["Lx"], spec["Nx"], spec["k0"], spec["kx0"])[0],
                dtype=complex)
            lam = lam[np.lexsort((lam.imag, lam.real))]
            arrays["strip_" + name] = np.asarray(
                eme_2d._ky_forward(lam, spec["qz2"]))
        out["strip"] = rows
        infinitesimal_pairs(eme_2d, out, arrays)
    if a.only in ("all", "rest"):
        mode_match_records(eme_diffraction, out)
        diffraction_records(eme_diffraction, out)
        vector_records(eme_2d_vector, out)
        split_forward_records(eme_2d_vector, out)
    out["secs"] = round(time.time() - t0, 1)
    if arrays:
        np.savez(HERE / "runs" / ("ve_ky_%s.npz" % a.tag), **arrays)
    _vh.dump(HERE / ("ve_run_%s.json" % a.tag), out)
    print("OK", a.build, a.tag, "in", out["secs"], "s")


if __name__ == "__main__":
    main()
