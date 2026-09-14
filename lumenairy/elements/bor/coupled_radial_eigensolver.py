"""BOR-PMM Milestone 2: coupled (E_r, E_phi) radial vector eigensolver.

Solves the cylindrical waveguide eigenproblem for eps = eps(r) (one azimuthal
order m, fields ~ exp(i m phi + i q z)) and returns the propagation constants q
and modal fields.  The three pieces that make it correct (each validated against
the open-cladding fiber oracle, ``fiber_oracle.py``):

1. **q^2 formulation with E_z elimination.**  Maxwell -> a generalized
   eigenproblem ``K Psi = q^2 B Psi`` in ``Psi = (E_r, E_phi)``, with the
   longitudinal ``E_z`` eliminated through ``Phi = (L_m + k0^2 eps)^{-1}[...]``
   (the cylindrical analog of the Cartesian PMM ``G = I - Kx(1/ezz)Kx``
   elimination in ``lumenairy/elements/pmm/_core.py:_sem_modes_tensor``).  This
   linear-in-``q^2`` form makes the solve a single dense ``eig``.

2. **Wall-normal inverse rule.**  At a ring interface ``D_r = eps E_r`` is
   continuous while ``E_r`` jumps; the normal-component eps uses the harmonic
   mean ([[1/eps]]^{-1}) rather than the pointwise value.  This removes the
   interface mode-doubling and sharpens the guided q to the oracle.  (Mirrors
   ``Cxx = [[1/exx]]^{-1}`` in the Cartesian solver; tangential eps stays
   pointwise.)

3. **Divergence-free filter.**  Real-space vector discretizations emit spurious
   modes that violate ``div(eps E) = 0``.  Physical modes have tiny relative
   divergence (~1e-2); spurious ones are O(1-10).  ``relative_divergence`` below
   flags them cleanly (>150x separation on the fiber test).

The radial operator uses a cell-centered grid (nodes at ``(i+1/2)h`` -> never
samples ``r = 0``, so the axis 1/r is regular; cf. the M1 Gauss-quadrature axis
treatment).  Convergence is 2nd-order in N (FD); a spectral-element upgrade is
the accuracy follow-on, but the FD form already matches the oracle to ~1e-4..1e-5.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.linalg import eig, lu_factor, lu_solve

from ._inv_census import census_inv


def _fd_grid(Rbig, N):
    """Cell-centered grid + 2nd-order differentiation matrix on (0, Rbig)."""
    h = Rbig / N
    r = (np.arange(N) + 0.5) * h
    D = np.zeros((N, N))
    for i in range(1, N - 1):
        D[i, i - 1] = -1.0 / (2 * h)
        D[i, i + 1] = 1.0 / (2 * h)
    D[0, 0] = -1.0 / h           # one-sided at the ends
    D[0, 1] = 1.0 / h
    D[N - 1, N - 2] = -1.0 / h
    D[N - 1, N - 1] = 1.0 / h
    return r, D, h


#: Where the staggered PEC wall is anchored (audit W6-B1).
#:
#: ``'rbig'`` (the DEFAULT) -- ``h = Rbig / (N + 0.5)``.  The outer stencil
#: closes the wall by forcing the tangential field to zero at the GHOST NODE
#: (index ``N``, radius ``(N + 0.5) h``), so THAT radius is the PEC wall; this
#: spacing lands it exactly on the ``Rbig`` the caller asked for, and the
#: scheme reaches the 2nd order this module documents (p = 1.99).
#:
#: ``'ghost'`` (LEGACY ESCAPE HATCH) -- ``h = Rbig / N``, which puts the ghost
#: node -- and therefore the wall -- at ``Rbig + h/2``: the discretized cavity
#: is half a cell LARGER than requested, and the box spectrum converges to
#: ``j_{m,n} / (Rbig + h/2)`` at FIRST order (p = 0.99).  It is a
#: KNOWN-DEFECTIVE anchor, not a supported alternative discretization; it
#: exists so the earlier numbers can be reproduced.
#:
#: The convergence tables behind both statements, the rejected
#: antisymmetric-ghost stencil and what the default costs on the
#: grazing-cutoff reproducer are in
#: ``docs/audits/MEASURE_BOR_STAGGERED_WALL_ANCHOR_2026_09_13.md``.
STAGGERED_WALL_ANCHOR = "rbig"


def _fd_grid_staggered(Rbig, N):
    """Yee-staggered (div-conforming) radial grid -- the spurious-mode CURE.

    Tangential ``E_phi, E_z`` (and ``Phi``) live on the cell-center NODES
    ``r_node = (i+1/2)h``; the wall-NORMAL ``E_r`` lives on the radial FACES
    ``r_face = (i+1)h`` where the flux ``D_r = eps E_r`` is single-valued.  The
    discrete de Rham identity ``curl . grad == 0`` then holds to machine
    precision, so the gradient null-space (the ~91% spurious sea of the nodal
    operator) collapses to a benign electrostatic branch at large ``|q^2|`` --
    OUT of the propagating window, with NO filtering/penalty and the eig size
    still ``2N`` (a complete square basis).  Returns the two grids + the four
    inter-grid bidiagonal operators (node->face / face->node derivative and
    average).

    LATTICE ANCHOR (audit W6-B1) -- see :data:`STAGGERED_WALL_ANCHOR`.  The
    outer stencil closes the PEC wall by forcing the tangential field to zero
    at the GHOST NODE (index ``N``, radius ``(N + 0.5) h``), so THAT radius is
    the wall.  The default ``'rbig'`` spacing ``h = Rbig / (N + 0.5)`` lands it
    exactly on ``Rbig`` and the scheme reaches the 2nd order this module
    documents (p = 1.99).  The legacy ``'ghost'`` spacing ``h = Rbig / N`` put
    the wall at ``Rbig + h/2`` -- half a cell OUTSIDE the requested domain --
    so the box spectrum converged to ``j_{m,n}/(Rbig + h/2)``, FIRST order
    (p = 0.99).
    """
    h = Rbig / N if STAGGERED_WALL_ANCHOR == "ghost" else Rbig / (N + 0.5)
    r_node = (np.arange(N) + 0.5) * h
    r_face = (np.arange(N) + 1.0) * h          # face i between node i, i+1
    Dn2f = np.zeros((N, N))                     # (Dn2f f)_{i+1/2} = (f_{i+1}-f_i)/h
    for i in range(N - 1):
        Dn2f[i, i] = -1.0 / h
        Dn2f[i, i + 1] = 1.0 / h
    Dn2f[N - 1, N - 1] = -1.0 / h               # outer wall: ghost node = 0 (PEC)
    Df2n = np.zeros((N, N))                     # (Df2n g)_i = (g_{i+1/2}-g_{i-1/2})/h
    for i in range(1, N):
        Df2n[i, i - 1] = -1.0 / h
        Df2n[i, i] = 1.0 / h
    Df2n[0, 0] = 1.0 / h                         # axis: innermost face only (g_{-1/2}=0)
    An2f = np.zeros((N, N))
    for i in range(N - 1):
        An2f[i, i] = An2f[i, i + 1] = 0.5
    An2f[N - 1, N - 1] = 0.5
    Af2n = np.zeros((N, N))
    for i in range(1, N):
        Af2n[i, i - 1] = Af2n[i, i] = 0.5
    Af2n[0, 0] = 0.5
    return r_node, r_face, h, Dn2f, Df2n, An2f, Af2n


#: the two radial boundary conditions the NODAL operators implement
_WALLS = ("natural", "pec")


def _check_wall(wall, staggered=False):
    """Validate the ``wall`` kwarg (audit W6-B3).  ``None`` means "the default
    for this basis": ``'natural'`` on the nodal path, the built-in closed
    for this basis".  An unrecognized value would otherwise fall
    through to ``'natural'`` silently, so a typo buys open-boundary physics
    (see docs/history/lumenairy.elements.bor.coupled_radial_eigensolver.md)."""
    if wall is None:
        return
    if wall not in _WALLS:
        raise ValueError(
            "wall must be one of %r or None (got %r)" % (list(_WALLS), wall))
    if staggered and wall != "pec":
        raise ValueError(
            "the staggered (Yee div-conforming) basis builds the closed "
            "Dirichlet wall into its node->face stencil; wall=%r would be "
            "IGNORED.  Pass wall='pec' (or omit it) on the staggered path, "
            "or staggered=False for the leaky natural wall." % (wall,))


def _normal_eps_faces(eps_node):
    """Wall-normal inverse-rule eps ON FACES: harmonic mean across the node-pair
    straddling each face (the inverse rule now lives where the jump physically
    is); pointwise inside a homogeneous ring."""
    n = len(eps_node)
    ef = np.asarray(eps_node, dtype=complex).copy()
    for i in range(n - 1):
        if eps_node[i] != eps_node[i + 1]:
            ef[i] = 2.0 / (1.0 / eps_node[i] + 1.0 / eps_node[i + 1])
    return ef


def _split_eps(raw, *, where="eps_profile"):
    """Accept an ISOTROPIC length-N eps array **or** an ``(N, 3)`` DIAGONAL
    tensor ``diag(eps_rr, eps_phiphi, eps_zz)`` in CYLINDRICAL components, and
    return the three per-node arrays.

    Anisotropy is restricted to the cylindrical-diagonal form because that is
    exactly the class a body of revolution can carry: an off-diagonal
    ``eps_r,phi`` (or a uniform Cartesian director) is NOT azimuthally
    invariant and couples the ``m`` harmonics, which this one-``m`` solver
    cannot represent.  Radially / azimuthally aligned uniaxial media ARE
    diagonal here and so are exactly representable.

    Isotropic input returns three references to the SAME array, so every
    assembled operator is byte-identical to the pre-anisotropy scalar path.
    """
    a = np.asarray(raw, dtype=complex)
    if a.ndim == 2 and a.shape[-1] == 3:
        return a[:, 0], a[:, 1], a[:, 2]
    if a.ndim == 1:
        return a, a, a
    raise ValueError(
        f"{where} must return either a length-N isotropic eps array or an "
        f"(N, 3) diagonal cylindrical tensor diag(eps_rr, eps_phiphi, "
        f"eps_zz); got array of shape {a.shape}.")


def _assemble_staggered(m, Rbig, N, eps_profile, k0):
    """Assemble the staggered q^2 generalized eigenproblem ``K Psi = q^2 B Psi``
    in ``Psi = (E_r[faces], E_phi[nodes])``.  Returns everything the mode/field
    recovery needs.  Operator placement (each term lands on its field's grid):
      E_r equation -> FACES,  E_phi equation -> NODES,  E_z/Phi -> NODES.
    """
    r_n, r_f, h, Dn2f, Df2n, An2f, Af2n = _fd_grid_staggered(Rbig, N)
    eps_rr, eps_pp, eps_zz = _split_eps(eps_profile(r_n))
    eps_node = eps_pp                    # TANGENTIAL (E_phi) -> pointwise
    eps_face = _normal_eps_faces(eps_rr)  # WALL-NORMAL (E_r) -> inverse rule
    diag = np.diag
    mrn, mrf = m / r_n, m / r_f
    A_n2f = Dn2f + diag(1.0 / r_f) @ An2f       # grad-like (d/dr+1/r): nodes -> faces
    A_f2n = Df2n + diag(1.0 / r_n) @ Af2n       # div-like:             faces -> nodes
    Lm = diag(1.0 / r_n) @ Df2n @ diag(r_f) @ Dn2f - diag(mrn ** 2)   # node Laplacian
    # E_z-elimination operand.  At a LONGITUDINAL resonance (k0^2 eps hits an
    # eigenvalue of -Lm) this matrix is near-singular and the inv() below --
    # which feeds Phi_r/Phi_p into BOTH K and B -- silently loses accuracy
    # (measured: cascade energy error jumps ~1e-14 -> ~1e-5..1e-4 in a
    # ~1e-11-relative k0 window).  Cheap LU pivot-ratio check + warn (audit
    # P3-13); the inv() itself is unchanged so off-resonance results are
    # byte-identical.
    Mz = Lm + k0 ** 2 * diag(eps_zz)      # LONGITUDINAL (E_z) component
    _du = np.abs(np.diag(lu_factor(Mz)[0]))
    if _du.size and _du.min() <= 1e-12 * _du.max():
        warnings.warn(
            f"_assemble_staggered: the E_z-elimination operator "
            f"Lm + k0^2*eps is near-singular (LU pivot ratio "
            f"{_du.min() / _du.max():.2e}) -- k0={k0:.10g} sits at a "
            f"longitudinal resonance of this (N={N}, Rbig={Rbig}) "
            f"discretization.  The modal basis loses ~8-10 digits here; "
            f"detune k0 by ~1e-8 relative (or change N) to restore "
            f"machine-precision energy.", stacklevel=2)
    Lei = census_inv(Mz, "coupled_radial.staggered:inv(Mz)")
    Phi_r = Lei @ (1j * A_f2n)                  # E_r[faces]  -> Phi[nodes]
    Phi_p = Lei @ (-diag(mrn))                  # E_phi[nodes]-> Phi[nodes]
    I = np.eye(N)
    B = np.block([[I + 1j * Dn2f @ Phi_r, 1j * Dn2f @ Phi_p],
                  [-diag(mrn) @ Phi_r,    I - diag(mrn) @ Phi_p]])
    K = np.block([[k0 ** 2 * diag(eps_face) - diag(mrf ** 2), -1j * diag(mrf) @ A_n2f],
                  [-1j * Df2n @ diag(mrf),  k0 ** 2 * diag(eps_node) + Df2n @ A_n2f]])
    return dict(K=K, B=B, Lei=Lei, A_f2n=A_f2n, Dn2f=Dn2f, mrn=mrn, mrf=mrf,
                eps_node=eps_node, eps_face=eps_face, eps_zz=eps_zz,
                r_n=r_n, r_f=r_f, h=h,
                Df2n=Df2n, An2f=An2f, Af2n=Af2n, k0=k0, m=m, N=N)


def _fast_geig(K, B):
    """Folded standard eigensolve of the pencil ``K x = q^2 B x`` -- ~2.6-3.4x
    faster than the generalized QZ ``eig(K, B)`` when ``B`` is well-conditioned
    (which it is here: ``B = I + Phi-coupling``, a mass-like matrix).

    Symmetric diagonal equilibration first (so the fold's conditioning matches
    the QZ's), then ``eig(Be^{-1} Ke)``.  An LU pivot-ratio guard (the pmm
    ``_core._fast_geig`` pattern, audit P3-13) routes NEAR-singular ``B`` --
    a longitudinal resonance -- to the robust generalized QZ ``eig(Ke, Be)``;
    LAPACK's ``solve`` raises only on an EXACT zero pivot, so an exception
    hook alone never fires there.  Note the guard protects only this fold: at
    an exact resonance the upstream ``Lei`` inversion in
    ``_assemble_staggered`` has already lost accuracy (it warns; detune k0 to
    restore machine precision).  Off-resonance, eigenvalues reproduce the QZ
    spectrum to ~1e-7 (the inaccuracy lands only in the deep-evanescent
    branch that decays in the cascade; the physical guided modes +
    machine-precision energy are unchanged -- validated)."""
    d = np.sqrt(np.abs(np.diag(B)))
    d = np.where(d > 0, 1.0 / d, 1.0)
    Ke = (d[:, None] * K) * d[None, :]
    Be = (d[:, None] * B) * d[None, :]
    lu, piv = lu_factor(Be)
    du = np.abs(np.diag(lu))
    if du.size and du.min() <= 1e-12 * du.max():   # near-singular -> robust QZ
        q2, z = eig(Ke, Be)
    else:
        q2, z = eig(lu_solve((lu, piv), Ke))
    return q2, d[:, None] * z


def _pec_wall_ops(D, h, N):
    """Apply a Dirichlet (PEC) wall at r=R to the cell-centered operators
    (M5a): the tangential field vanishes at the wall via the antisymmetric ghost
    ``f_N = -f_{N-1}`` (the wall sits halfway between node N-1 and ghost N).
    Returns ``(Dd, Lap)`` -- the wall-corrected first derivative and the
    Dirichlet Laplacian ``d^2/dr^2`` -- turning the leaky natural-wall
    propagating modes into a clean real-q box spectrum (validated: same-medium
    identity ~3e-12, multi-mode Fresnel mean ~2e-3).  Axis row 0 is untouched
    (the cell-centered grid never samples r=0)."""
    Dd = D.copy()
    Dd[N - 1, N - 2] = -1.0 / (2 * h)         # (f_N - f_{N-2})/2h, f_N=-f_{N-1}
    Dd[N - 1, N - 1] = -1.0 / (2 * h)
    Lap = np.zeros((N, N), dtype=D.dtype)
    for i in range(1, N - 1):
        Lap[i, i - 1] = 1.0 / h ** 2
        Lap[i, i] = -2.0 / h ** 2
        Lap[i, i + 1] = 1.0 / h ** 2
    Lap[0, 0] = -1.0 / h ** 2                  # axis: regular (Neumann-like)
    Lap[0, 1] = 1.0 / h ** 2
    Lap[N - 1, N - 2] = 1.0 / h ** 2           # wall ghost: -3 on the diagonal
    Lap[N - 1, N - 1] = -3.0 / h ** 2
    return Dd, Lap


def _normal_eps(eps):
    """Wall-normal effective eps: harmonic mean across each eps jump
    ([[1/eps]]^{-1} inverse rule), pointwise elsewhere.

    ORDER INDEPENDENCE (audit W6-B9).  The historical loop wrote the pair
    ``en[i] = en[i-1] = 1/hm`` IN PLACE, so a node sitting between two
    back-to-back jumps (a ring exactly ONE node wide) had the inner
    interface's harmonic mean silently CLOBBERED by the outer one -- the
    normal eps then depended on the loop direction, i.e. mirroring the radial
    profile did not mirror the operator (measured: ``[2,6,3,3] -> [3,4,4,3]``
    but the mirrored profile gave ``[3,4,3,3]`` reversed, not ``[3,4,4,3]``).
    Each interface's inverse-eps mean is now accumulated and AVERAGED over the
    interfaces a node participates in, which is mirror-symmetric by
    construction and BIT-IDENTICAL for every isolated interface (rings two or
    more nodes wide -- i.e. every profile the gates exercise).
    """
    inv = (1.0 / eps).copy()
    acc = np.zeros(len(eps), dtype=inv.dtype)
    hits = np.zeros(len(eps), dtype=int)
    for i in range(1, len(eps)):
        if eps[i] != eps[i - 1]:
            # same expression as the historical `1.0 / (2.0 / (a + b))` so an
            # isolated interface stays bit-identical
            hmi = 1.0 / (2.0 / (inv[i] + inv[i - 1]))
            acc[i] += hmi
            acc[i - 1] += hmi
            hits[i] += 1
            hits[i - 1] += 1
    en = np.where(hits > 0, acc / np.maximum(hits, 1), inv)
    return 1.0 / en


def _pml_stretch(r, h, R_pml, Rbig, sigma_max, p):
    """Complex coordinate stretch for the radial PML (M3): ``s = 1 + i sigma``
    with ``sigma`` ramping polynomially in ``[R_pml, Rbig]``.  Returns
    ``sinv = 1/s`` and the stretched coordinate ``rt = int_0^r s dr'`` (== r in
    the physical region).  Outgoing cylindrical waves are absorbed; bound modes
    (decayed before R_pml) are untouched (verified: q invariant in sigma_max)."""
    t = np.clip((r - R_pml) / (Rbig - R_pml), 0.0, 1.0)
    s = 1.0 + 1j * sigma_max * t ** p
    rt = np.cumsum(s) * h - 0.5 * s * h
    return 1.0 / s, rt


def _mode_reldiv(Er, Ephi, qj, D, mr, Lei, A, eps, eps_n, rg, k0, *,
                 eps_z=None):
    """Longitudinal field ``Ez`` and the relative-divergence diagnostic for ONE
    nodal radial vector mode (E_z-eliminated form).

    THE single source of the nodal ``reldiv`` formula: used both here (the
    eigensolver's mode harvest) and by ``zcascade.layer_modes(with_reldiv=True)``
    so that ``bor_solve.build_layer(basis='nodal')`` derives the divergence tag
    from the SAME dense eig it already ran for the modal basis, instead of a
    second byte-identical ``eig(K, B)`` (audit AUDIT_V5_24_2 S1-18).

    ``reldiv`` is invariant under ``qj -> -qj`` (the forward-orientation flip):
    ``Ez`` scales with ``qj`` and enters the divergence only as ``qj*Ez`` (i.e.
    ``qj**2``) while ``|Ez|`` enters the norm, so the caller may pass either the
    raw ``sqrt(q2)`` root or the forward-oriented root and get the byte-identical
    value.  Returns ``(Ez, reldiv)`` -- ``Ez`` is handed back so the eigensolver
    does not recompute it for the mode dict."""
    Ez = qj * (Lei @ (1j * A @ Er - mr @ Ephi))
    Dr = eps_n * Er
    _ez = eps if eps_z is None else eps_z
    div = (1.0 / rg) * (D @ (rg * Dr)) + 1j * mr @ (eps * Ephi) + 1j * qj * (_ez * Ez)
    En = np.sqrt(np.sum(np.abs(Er) ** 2 + np.abs(Ephi) ** 2 + np.abs(Ez) ** 2))
    reldiv = np.sqrt(np.sum(np.abs(div) ** 2)) / (k0 * max(En, 1e-300))
    return Ez, float(reldiv.real)


def radial_coupled_modes(m, Rbig, N, eps_profile, k0, *, inverse_rule=True,
                         R_pml=None, sigma_max=5.0, pml_p=2, wall=None,
                         staggered=False):
    """All radial vector modes for azimuthal order ``m``.

    ``eps_profile`` : callable r-array -> eps-array (real or complex).
    ``R_pml`` : if set, a radial PML (M3) occupies ``[R_pml, Rbig]`` so the
    radial boundary is OPEN (radiation modes absorbed -> complex q; bound modes
    unchanged).  Default ``None`` = hard wall (byte-identical to the M2 path).
    ``wall`` : ``'natural'`` (default) = the leaky one-sided wall; ``'pec'`` =
    the Dirichlet (closed) wall.  ``staggered`` : if True, use the Yee
    div-conforming discretization (the spurious-mode CURE) -- ``E_r`` on faces,
    ``E_phi`` on nodes; the spurious gradient sea collapses out of the physical
    window (eig still 2N).  Returns a list of dicts: ``q``, ``reldiv``,
    ``Er``/``Ephi``/``Ez`` fields, ``r``.

    VALIDATION (audit W6-B3/B4).  ``wall`` is now checked against the two
    supported values: an unrecognized value (a typo such as ``'PEC'``) used to
    fall through to the leaky ``'natural'`` wall SILENTLY, so a caller asking
    for a closed box quietly got open-boundary physics.  And the staggered
    path -- which builds the closed Dirichlet wall into its node->face stencil
    and always applies the FACE inverse rule -- now rejects the nodal-only
    ``R_pml`` / ``wall='natural'`` / ``inverse_rule=False`` instead of
    silently ignoring them (measured: bit-identical output, i.e. a caller
    asking for an OPEN radial boundary got the closed wall with no signal).
    """
    _check_wall(wall, staggered)
    if staggered:
        if R_pml is not None:
            raise ValueError(
                "radial_coupled_modes(staggered=True): the radial PML is a "
                "NODAL-basis feature; the staggered (Yee div-conforming) "
                "discretization builds in the closed Dirichlet wall and would "
                "IGNORE R_pml=%r.  Use staggered=False for an open radial "
                "boundary." % (R_pml,))
        if not inverse_rule:
            raise ValueError(
                "radial_coupled_modes(staggered=True): inverse_rule=False is "
                "not available on the staggered basis -- the wall-normal "
                "inverse rule lives on the FACE grid there by construction "
                "(_normal_eps_faces) and cannot be switched off.")
        return _radial_coupled_modes_staggered(m, Rbig, N, eps_profile, k0)
    wall = "natural" if wall is None else wall
    r, D, h = _fd_grid(Rbig, N)
    eps_rr, eps, eps_zz = _split_eps(eps_profile(r))   # eps = tangential (E_phi)
    eps_n = _normal_eps(eps_rr) if inverse_rule else eps_rr
    Lap = None
    if R_pml is not None:
        sinv, rg = _pml_stretch(r, h, R_pml, Rbig, sigma_max, pml_p)
        D = np.diag(sinv) @ D                 # d/dr_tilde (stretched)
    else:
        rg = r                                # rg = the (possibly stretched) radius
        if wall == "pec":                     # M5a: Dirichlet (closed) wall
            D, Lap = _pec_wall_ops(D, h, N)
    I = np.eye(N)
    ir = np.diag(1.0 / rg)
    mr = m * ir
    m2r2 = (m ** 2) * np.diag(1.0 / rg ** 2)
    A = D + ir
    Lm = (D @ D if Lap is None else Lap) + ir @ D - m2r2
    dA = D @ A
    # E_z elimination
    Lei = census_inv(Lm + k0 ** 2 * np.diag(eps_zz),
                     "coupled_radial.nodal:Ez_elimination")
    Phi_r = Lei @ (1j * A)
    Phi_p = Lei @ (-mr)
    B = np.block([[I + 1j * D @ Phi_r, 1j * D @ Phi_p],
                  [-mr @ Phi_r,        I - mr @ Phi_p]])
    K = np.block([[k0 ** 2 * np.diag(eps_n) - m2r2, -1j * mr @ A],
                  [-1j * D @ mr,        k0 ** 2 * np.diag(eps) + dA]])
    q2, Vm = eig(K, B)
    q = np.sqrt(q2)
    modes = []
    for j in range(len(q)):
        Er = Vm[:N, j]
        Ephi = Vm[N:, j]
        # div(eps E) via the shared ``_mode_reldiv`` helper, using the CONSISTENT
        # normal flux D_r = eps_n E_r (the same inverse-rule eps the operator
        # uses) and pointwise eps on the tangential components.  (Using pointwise
        # eps for D_r instead inflates the physical modes' divergence ~100x and
        # breaks the spurious/physical separation.)
        Ez, reldiv = _mode_reldiv(Er, Ephi, q[j], D, mr, Lei, A, eps, eps_n,
                                  rg, k0, eps_z=eps_zz)
        modes.append(dict(q=q[j], reldiv=reldiv,
                          Er=Er, Ephi=Ephi, Ez=Ez, r=r))
    return modes


def _radial_coupled_modes_staggered(m, Rbig, N, eps_profile, k0):
    """Staggered (div-conforming) modes -- the spurious-mode cure.  ``Er`` on
    faces, ``Ephi``/``Ez`` on nodes; the divergence diagnostic is exact-by-
    construction (flux ``D_r = eps_face Er`` lives on the face)."""
    op = _assemble_staggered(m, Rbig, N, eps_profile, k0)
    q2, Vm = _fast_geig(op["K"], op["B"])
    q = np.sqrt(q2)
    Lei, A_f2n, Df2n = op["Lei"], op["A_f2n"], op["Df2n"]
    mrn, eps_node, eps_face = op["mrn"], op["eps_node"], op["eps_face"]
    eps_zz = op["eps_zz"]
    r_n, r_f = op["r_n"], op["r_f"]
    modes = []
    for j in range(len(q)):
        Er, Ephi = Vm[:N, j], Vm[N:, j]            # Er on faces, Ephi on nodes
        Ez = q[j] * (Lei @ (1j * A_f2n @ Er - mrn * Ephi))     # nodes
        Dr = eps_face * Er                          # face flux (native-continuous)
        div = ((1.0 / r_n) * (Df2n @ (r_f * Dr)) + 1j * mrn * (eps_node * Ephi)
               + 1j * q[j] * (eps_zz * Ez))         # all on nodes
        En = np.sqrt(np.sum(np.abs(Er) ** 2 + np.abs(Ephi) ** 2 + np.abs(Ez) ** 2))
        reldiv = np.sqrt(np.sum(np.abs(div) ** 2)) / (k0 * max(En, 1e-300))
        modes.append(dict(q=q[j], reldiv=float(reldiv.real),
                          Er=Er, Ephi=Ephi, Ez=Ez, r=r_n))
    return modes


#: Samples the guided-mode ROOT CENSUS puts across the guided window
#: (:func:`_step_index_root_census`).  Cost is one 4x4 Bessel determinant per
#: sample: MEASURED 49-72 ms for 2001 samples, against 0.72 s (N = 150),
#: 17.9 s (N = 400) and 67.8 s (N = 600) for the FD eigensolve it guards --
#: i.e. 9.9% of the call at the smallest grid anyone uses and 0.1-0.4% at the
#: grids real work runs.  ``census=False`` removes it entirely.
#:
#: RESOLUTION, and the FAILURE MODE it buys.  The census counts SIGN CHANGES of
#: the exact hybrid determinant on a uniform ``n_eff`` grid, so it resolves two
#: roots only if they are more than one cell -- ``(n_core - n_clad) / 2000`` in
#: ``n_eff`` -- apart.  Two roots closer than that (a near-degenerate pair, e.g.
#: the HE/EH partners of the same LP group at weak contrast, or a TANGENTIAL
#: double root) are counted ONCE or not at all.  That error is one-sided in the
#: safe direction: the census can only UNDER-count, so the notice below can miss
#: a genuine shortfall but can never invent one.  MEASURED against the
#: bisecting :func:`~.fiber_oracle.fiber_modes` on the fixtures in
#: ``test_audit2609_a14_verify.py``: identical counts on every one.
_CENSUS_SCAN = 2001


def _step_index_root_census(m, a, eps_core, eps_clad, k0,
                            n_scan=_CENSUS_SCAN):
    """How many guided roots the EXACT step-index hybrid characteristic
    equation has for azimuthal order ``m`` -- ``None`` when no census is
    possible (a lossy or inverted profile, or non-numeric permittivities).

    This is an INDEPENDENT count: :func:`~.fiber_oracle.fiber_det` is a 4x4
    boundary-match determinant of Bessel functions that shares no code with the
    finite-difference vector eigensolver :func:`radial_coupled_modes` whose
    output it is compared against.  Only SIGN CHANGES are counted -- no
    bisection -- so the cost is one determinant per sample; see
    ``_CENSUS_SCAN`` for the resolution and its one-sided failure mode.

    The scan interval excludes both edges by ``1e-7`` relative (the same margin
    :func:`~.fiber_oracle.fiber_modes` uses), where ``1/g1^2`` and ``1/kap^2``
    in the determinant diverge; in between every entry is finite and smooth, so
    a sign change is a root and not a pole.
    """
    try:
        e1, e2 = complex(eps_core), complex(eps_clad)
    except (TypeError, ValueError):
        return None                        # traced / array-valued: no census
    if (abs(e1.imag) > 1e-12 * max(abs(e1), 1.0)
            or abs(e2.imag) > 1e-12 * max(abs(e2), 1.0)):
        return None                        # lossy: a real-axis scan is no census
    if not (e1.real > e2.real > 0.0):
        return None                        # inverted / non-guiding profile
    from .fiber_oracle import fiber_det
    lo = np.sqrt(e2.real) * k0 * (1.0 + 1e-7)
    hi = np.sqrt(e1.real) * k0 * (1.0 - 1e-7)
    if not hi > lo:
        return None
    qs = np.linspace(lo, hi, int(n_scan))
    try:
        d = np.array([float(np.real(fiber_det(q, int(m), a, e1.real, e2.real,
                                              k0))) for q in qs])
    except Exception:                      # noqa: BLE001 - never break the solve
        return None
    good = np.isfinite(d)
    n_roots, prev = 0, -1
    for i in range(qs.size):
        if not good[i]:
            prev = -1
            continue
        if prev >= 0 and (d[i] == 0.0 or d[prev] * d[i] < 0.0):
            n_roots += 1
        prev = i
    return n_roots


def guided_modes(m, a, Rbig, N, eps_core, eps_clad, k0, *,
                 reldiv_tol=1.0, tail_tol=0.05, census=True):
    """Bound guided modes (div-free AND decaying in the cladding), q descending.

    UNIT INVARIANCE (audit W6-B2).  The guided-window margin and the real-axis
    tolerance are expressed as fractions of ``k0``, not as absolute numbers on
    ``q``.  ``q`` has units 1/length, so the historical absolute ``1e-2`` /
    ``1e-3`` were an implicit "lengths are microns" assumption: the SAME
    physical fiber written in nanometres (``k0 = 2e-3`` /nm) has its entire
    guided window ``sqrt(eps_clad) k0 .. sqrt(eps_core) k0`` = 2.8e-3..4.9e-3
    BELOW the 1e-2 margin, so ``qlo + 1e-2 < q < qhi - 1e-2`` was empty and
    this function silently returned ``[]`` while the raw spectrum held the
    correct bound modes.

    CONTRAST INVARIANCE (audit H1, 2026-09-12) is the SECOND half of that same
    statement and it was missing.  Making the margin a fraction of ``k0``
    (``5e-3 * k0`` per side) left it a fraction of the WRONG quantity: the
    guided window is ``(n_core - n_clad) * k0`` wide, so a ``5e-3 * k0`` band
    on EACH side admits nothing at all unless ``n_core - n_clad > 0.01``.  Real
    fibers live an order of magnitude below that -- telecom SMF is
    ``dn ~ 0.005``, and the textbook ``V = 2.4`` fiber (1.45 / 1.44) has
    ``dn = 0.01`` exactly, collapsing the admissible interval to a single
    point.  Both returned ``[]`` while ``radial_coupled_modes`` held the correct
    HE11 (measured ``n_eff = 1.445293453`` against the exact hybrid oracle's
    ``1.445293173`` at ``N = 300``), i.e. the eigensolver was right and the
    FILTER threw the answer away -- and the empty list is indistinguishable
    from a genuinely cut-off structure.  The margin is now a fraction of the
    WINDOW, ``max(1e-6 * k0, 1e-3 * (qhi - qlo))``, which is invariant under
    BOTH a unit rescale and an index-contrast rescale; the ``1e-6 * k0`` floor
    keeps a band on a window so narrow that ``1e-3`` of it would be rounding.
    The old ``5e-3 * k0`` is reproduced only where it was ever meaningful
    (``dn = 5`` would be needed for the two to coincide), so every fixture with
    a realistic contrast changes from "no modes" to "the modes that are there".

    Raises ``ValueError`` when the window is narrower than the guard band it
    would need (``qhi - qlo <= 2 * q_margin``, i.e. ``dn <= 2e-6``): nothing can
    be admitted there, and saying so is the one thing an empty list cannot.
    Warns when the filters empty a window that DID hold candidates, naming the
    closest one AND the filter that rejected it, so the "silent []" failure
    mode cannot come back through a different door.  There are THREE such doors
    -- the guard band, the ``reldiv_tol`` spurious-mode screen and the
    ``tail_tol`` radiation screen -- and only the band is about contrast: on a
    HIGH-contrast fibre (Si/SiO2, ``dn = 2.04``, V = 2.0) and on any fibre a
    little above its next mode's cut-off, it is the TAIL screen that empties
    the list, because ``Rbig`` is too small for the mode's cladding tail.  An
    empty list is therefore never by itself evidence that a structure does not
    guide; the warning says which knob to turn.

    A PARTIAL result is audible too (``census=True``, the default).  An empty
    list is the loud case; the quiet one is a list that is merely SHORT --
    measured on Si/SiO2 (``dn = 2.04``) at V = 4.0, where this returned ONE
    mode while the exact hybrid characteristic equation has THREE
    (``n_eff`` = 3.038391486 / 1.566626407 / 1.440009396).  Every call now also
    counts the roots of that exact equation for the requested ``m``
    (:func:`_step_index_root_census` -- a sign-change scan of the 4x4 Bessel
    determinant, independent of the finite-difference eigensolver) and WARNS,
    naming both counts and the order, when the solver returns fewer.  The
    census can only UNDER-count (see ``_CENSUS_SCAN``), so the notice never
    fires spuriously.  ``census=False`` skips the scan for a hot loop that has
    already established its resolution.
    """
    def eps_profile(rr):
        return np.where(rr <= a, eps_core, eps_clad)
    out = []
    qlo, qhi = np.sqrt(eps_clad) * k0, np.sqrt(eps_core) * k0
    window = float(np.real(qhi - qlo))
    # Relative to the WINDOW (contrast-invariant), floored on k0 (unit-invariant).
    q_margin = max(1e-6 * k0, 1e-3 * window)
    # Unchanged at every contrast the historical value was meaningful at; the
    # clamp only binds when 5e-4*k0 exceeded half the window itself, where the
    # old tolerance admitted a q whose imaginary part was larger than the whole
    # guided band.
    imag_tol = min(5e-4 * k0, 0.5 * window)
    if window <= 2.0 * q_margin:
        raise ValueError(
            f"guided_modes: the guided window is empty -- "
            f"sqrt(eps_core) - sqrt(eps_clad) = {window / k0:.3e} (in units of "
            f"k0) is not wider than the 2 x {q_margin / k0:.3e} guard band the "
            f"filter needs, so no q can be admitted whatever the spectrum "
            f"holds.  Increase the index contrast (eps_core = {eps_core!r}, "
            f"eps_clad = {eps_clad!r}) or call radial_coupled_modes directly "
            f"and classify the raw spectrum yourself.")
    # Closest candidate INSIDE the guided window that some filter rejected, and
    # WHICH filter rejected it.  Every rejecting filter is recorded, not just
    # the guard band: the band is one of three doors an empty list can come out
    # of, and the other two (the reldiv spurious-mode screen and the tail
    # radiation screen) are the ones that fire on a HIGH-contrast or
    # near-cut-off fibre -- measured on Si/SiO2 (dn = 2.04) at V = 2.0, where
    # the raw spectrum holds n_eff = 1.882 against the exact hybrid oracle's
    # 1.8469 and the TAIL screen rejects it, and on SiN/SiO2 at V = 2.0, where
    # the raw mode is 4.6e-05 from the exact HE11 and is rejected the same way.
    # Warning on the band alone would have left those silent, which is the
    # failure mode this notice exists to remove.
    near = None                       # (mode, why) for the closest rejection

    def _note(md, why):
        nonlocal near
        if near is None or md["q"].real > near[0]["q"].real:
            near = (md, why)

    for md in radial_coupled_modes(m, Rbig, N, eps_profile, k0):
        q = md["q"]
        if not (qlo < q.real < qhi and abs(q.imag) < imag_tol):
            continue
        if not (qlo + q_margin < q.real < qhi - q_margin):
            _note(md, f"it sits inside the {q_margin / k0:.3e} k0 guard band "
                      f"at the edge of the guided window (too close to "
                      f"cut-off for this filter)")
            continue
        if md["reldiv"] > reldiv_tol:
            _note(md, f"its relative divergence {md['reldiv']:.2e} exceeds "
                      f"reldiv_tol = {reldiv_tol:g} (the SPURIOUS-mode "
                      f"screen)")
            continue                                    # spurious
        amp = np.abs(md["Er"]) + np.abs(md["Ephi"])
        amp = amp / amp.max()
        tail = float(amp[md["r"] > 0.8 * Rbig].max())
        if tail > tail_tol:
            _note(md, f"its field amplitude beyond r = 0.8 Rbig is "
                      f"{tail:.2e}, above tail_tol = {tail_tol:g} (the "
                      f"RADIATION screen) -- Rbig is most likely too small "
                      f"for this mode's cladding tail")
            continue                                    # radiation, not bound
        out.append(md)
    if not out and near is not None:
        md, why = near
        warnings.warn(
            f"guided_modes: no mode survived the filters, but the raw spectrum "
            f"holds a mode INSIDE the guided window "
            f"{qlo.real / k0:.9f}..{qhi.real / k0:.9f} at n_eff = "
            f"{md['q'].real / k0:.9f} (reldiv = {md['reldiv']:.2e}), and it "
            f"was rejected because {why}.  An empty list here means 'no mode "
            f"passed THIS filter', NOT 'no guided mode': refine N / Rbig, "
            f"relax reldiv_tol / tail_tol, or inspect radial_coupled_modes "
            f"directly.", stacklevel=2)
    if census:
        n_exact = _step_index_root_census(m, a, eps_core, eps_clad, k0)
        if n_exact is not None and len(out) < n_exact:
            warnings.warn(
                f"guided_modes: returned {len(out)} mode(s) for m = {m}, but "
                f"the EXACT step-index hybrid characteristic equation has "
                f"{n_exact} root(s) in the guided window "
                f"{np.sqrt(np.real(eps_clad)):.9f}..{np.sqrt(np.real(eps_core)):.9f} "
                f"(n_eff) at this m -- {n_exact - len(out)} mode(s) of the "
                f"exact spectrum did not survive the filters.  The census is a "
                f"sign-change scan of an INDEPENDENT 4x4 Bessel determinant and "
                f"can only under-count, so this shortfall is real: the usual "
                f"cause is Rbig too small for the weakly-bound modes' cladding "
                f"tails (raise Rbig, then N), or reldiv_tol / tail_tol too "
                f"tight.  Pass census=False to silence the scan.",
                stacklevel=2)
    return sorted(out, key=lambda md: -md["q"].real)
