"""BOR-PMM: the ``BORStack`` public-API prototype (axisymmetric / body-of-
revolution stack), mirroring ``lumenairy.elements.pmm.PMMStack`` but in
cylindrical coordinates.

A stack of z-layers, each with a radial permittivity profile ``eps(r)`` at one
azimuthal order ``m`` (fields ~ ``exp(i m phi + i q z)``).  Built on the
spurious-free Yee div-conforming radial solver (``staggered=True``), so the
modal basis is complete + spurious-free and the cascade conserves energy to
machine precision.

Example
-------
    s = BORStack(Rbig=4.0, m=1, N=300, n_superstrate=1.4142, n_substrate=1.4142)
    s.add_layer(0.5, rings=(0.8, 0.5, 2.449, 1.414))   # period, duty, n_ridge, n_groove
    s.set_source(wavelength=2*pi/2.0)
    res = s.solve()        # res['R'], res['T'] per propagating order, res['angles']
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np

from ...backend import is_jax_array as _is_jax_array
from .zcascade import interface_smatrix, layer_modes, propagation_smatrix, redheffer_star

# v5.17.1 (audit P3-12): bound on the per-instance modal-basis LRU.  Each
# entry holds one layer's 2N x 2N modal matrices (W, V ~ 5.8 MB each at the
# default N=300), so 16 slots cap the cache at ~190 MB worst-case while
# still covering a full stack of distinct profiles plus a couple of swept
# k0 points.  Wave-4 LRU conventions: OrderedDict + move_to_end on hit +
# popitem(last=False) eviction.
_MODAL_CACHE_SIZE = 16


def _sem_flux(W_E, V_H, wq_face, wq_node, n1):
    """z-Poynting flux of an SEM-basis field, one value per column.  Row
    layout differs from the FD staggered contract: W rows [:n1] = E_r (V1),
    [n1:] = E_phi (V0); V rows [:n0p] = h_r (V0), [n0p:] = h_phi (V1) --
    each h component on its flux-partner grid, so the pairing is
    E_r (x) h_phi on the V1 weights and E_phi (x) h_r on the V0 weights."""
    n0p = W_E.shape[0] - n1
    return np.real(
        np.sum(W_E[:n1] * np.conj(V_H[n0p:]) * wq_face[:, None], axis=0)
        - np.sum(W_E[n1:] * np.conj(V_H[:n0p]) * wq_node[:, None], axis=0))


def _column_phases(W):
    """Deterministic per-column phase of a modal basis: the phase of each
    column's largest-magnitude entry.  Dividing a column by its phase pins
    the LAPACK-arbitrary eigenvector gauge (AUDIT_DYNAMETA_CONSUMER_API_GAPS
    C1a): the pinned convention is 'the dominant field sample of every mode
    profile is real-positive'."""
    idx = np.argmax(np.abs(W), axis=0)
    piv = W[idx, np.arange(W.shape[1])]
    mag = np.abs(piv)
    return np.where(mag > 0, piv / np.where(mag > 0, mag, 1.0), 1.0)


def _stag_flux(W_E, V_H, wq_face, wq_node):
    """Total z-Poynting flux of a staggered-basis field, one value per
    column: ``Re(sum Er conj(hphi) wq_face - sum Ephi conj(hr) wq_node)``
    (the two-grid quadrature -- Er/hphi live on FACES, Ephi/hr on NODES;
    audit P3-14)."""
    N = len(wq_face)
    return np.real(
        np.sum(W_E[:N] * np.conj(V_H[N:]) * wq_face[:, None], axis=0)
        - np.sum(W_E[N:] * np.conj(V_H[:N]) * wq_node[:, None], axis=0))


class BORStack:
    """Axisymmetric (body-of-revolution) modal stack solver.

    The cylindrical-coordinate peer of :class:`~lumenairy.RCWAStack` /
    :class:`~lumenairy.PMMStack` / :class:`~lumenairy.BerremanStack`;
    top-level exported as ``lumenairy.BORStack`` since v5.25 (audit
    S5-10 / B2).

    Terminology (mode vs order)
    ---------------------------
    These two words are used deliberately and consistently across the BOR
    API (audit S5-10 / B2 -- previously mixed):

    * **azimuthal order** ``m`` -- the integer harmonic index of the
      ``exp(i m phi)`` azimuthal dependence fixed at construction.  The
      whole stack is solved at one ``m``.  (This is the cylindrical analog
      of a Cartesian Fourier-harmonic index, NOT a diffraction channel.)
    * **order** -- a propagating diffraction channel of the cascade: an
      entry of the ``q`` / ``gamma`` / ``angles`` / ``R`` / ``T`` /
      ``energy`` arrays returned by :meth:`solve`.  Orders are the
      cylindrical peer of the RCWA/PMM planar diffraction orders and index
      the *observable* per-channel efficiencies.
    * **mode** -- a column of the radial modal basis (an eigenvector
      ``(W, V)`` of the staggered radial operator), i.e. the basis in
      which the S-matrix ``S`` is expressed.  Every propagating order maps
      to one propagating mode; :meth:`per_mode_amplitudes` reports modal
      amplitudes in a pinned deterministic gauge.

    Result container (S5-6 forward-compat)
    --------------------------------------
    :meth:`solve` currently returns a plain ``dict`` (keys ``q``,
    ``gamma``, ``angles``, ``R``, ``T``, ``energy``, ``S``).  The still-open
    S5-6 work item unifies BOR / RCWA / PMM / Berreman onto ONE structured
    solve-result container.  That migration is intentionally NOT taken here
    (it is a breaking return-shape change and must land together with the
    other engines, not churn BOR alone -- see audit S5-10 / B2).  When it
    lands, the structured container will remain **mapping-compatible**
    (``res['R']`` / ``res['T']`` keep working) so this dict contract is a
    forward-compatible alias, not a hard break.
    """

    def __init__(self, Rbig, m, *, n_substrate=1.0, n_superstrate=1.0, N=300,
                 basis="fd", degree=8, elements_per_segment=1,
                 grade=True, n_mesh_cap=None):
        # Input validation (audit P3-10) -- mirror the PMMStack/PMM2DStack
        # builder guards: a bad domain/grid propagates to plausible-looking
        # garbage (negative-radius grid, 1-point operators) deep in the build.
        Rbig = float(Rbig)
        if not np.isfinite(Rbig) or Rbig <= 0.0:
            raise ValueError(
                f"BORStack: Rbig (domain radius) must be > 0, got {Rbig}.")
        if not float(m).is_integer():
            raise ValueError(
                f"BORStack: m (azimuthal order) must be an integer, got {m!r}.")
        if not float(N).is_integer() or int(N) < 2:
            raise ValueError(
                f"BORStack: N (radial grid points) must be an integer >= 2, "
                f"got {N!r}.")
        # Half-space index validation (audit W6-B6 -- the P3-10 sibling gap).
        # These were the only builder arguments left unguarded, and each
        # failure mode is silent or cryptic: n = 0 gives eps = 0, an empty
        # propagating set and hence an EMPTY R/T (exactly the failure P3-10
        # guarded for wavelength <= 0); a NEGATIVE n silently means |n| because
        # only n**2 is used (measured: n_superstrate=-1.5 solved happily with 5
        # orders); NaN/inf reach LAPACK as "array must not contain infs or
        # NaNs" / OverflowError from deep inside the eigensolve.
        for _nm, _n in (("n_superstrate", n_superstrate),
                        ("n_substrate", n_substrate)):
            _nc = complex(_n)
            if not (np.isfinite(_nc.real) and np.isfinite(_nc.imag)):
                raise ValueError(
                    f"BORStack: {_nm} must be finite, got {_n!r}.")
            if _nc.real <= 0.0:
                raise ValueError(
                    f"BORStack: {_nm} must have a positive real part (a "
                    f"refractive index), got {_n!r}.  Only n**2 is used, so a "
                    f"negative n would silently mean |n| and n = 0 gives an "
                    f"empty propagating set (empty R/T).")
        if basis not in ("fd", "sem"):
            raise ValueError(
                "BORStack: basis must be 'fd' (Yee-staggered finite "
                "difference, the default) or 'sem' (div-conforming "
                f"spectral elements), got {basis!r}.")
        degree = int(degree)
        if degree < 2:
            raise ValueError(
                f"BORStack: degree must be >= 2, got {degree}.")
        eps_seg = int(elements_per_segment)
        if eps_seg < 1:
            raise ValueError(
                f"BORStack: elements_per_segment must be >= 1, got "
                f"{elements_per_segment}.")
        # n_mesh_cap: concrete upper bound on |n| of TRACED (JAX) segment eps,
        # used only by the SEM wavelength-resolution cap, whose element sizing
        # needs a numeric refractive index that a traced value cannot provide
        # under jax.grad/jit.  An UPPER bound is safe by construction (the mesh
        # can only over-resolve).  Required as soon as any traced eps reaches a
        # basis='sem' solve; ignored otherwise.
        if n_mesh_cap is not None:
            n_mesh_cap = float(n_mesh_cap)
            if not np.isfinite(n_mesh_cap) or n_mesh_cap <= 0.0:
                raise ValueError(
                    f"BORStack: n_mesh_cap must be > 0, got {n_mesh_cap}.")
        self.n_mesh_cap = n_mesh_cap
        self.basis = basis
        self.degree = degree
        self.elements_per_segment = eps_seg
        self.grade = bool(grade)
        self.Rbig = Rbig
        self.m = int(m)
        self.N = int(N)
        self.eps_sub = complex(n_substrate) ** 2
        self.eps_sup = complex(n_superstrate) ** 2
        self._layers = []   # list of (thickness, eps_profile callable, profile key)
        # SEM sidecar per layer: (interior_walls tuple, segments tuple) --
        # segments = ((r_out, eps_triple), ...) covering (0, Rbig]; None for
        # specs the SEM basis cannot mesh (bare profile callables carry no
        # wall information).
        self._sem_layers = []
        # Parallel differentiable spec per layer: (thickness, jbuild(r_n, jnp) ->
        # jnp eps_node, raw_values) -- lets the JAX twin rebuild eps_node in the
        # active namespace from the RAW (possibly-traced) ring index / eps.
        self._jax_layers = []
        self.k0 = None
        # retained state of the last NumPy solve (AUDIT_DYNAMETA_CONSUMER_API_
        # GAPS C1): _last feeds per_mode_amplitudes (always retained -- cheap
        # references into the modal LRU); _internal holds the partial cascades
        # for layer_absorption (retain_internal=True only).
        self._last = None
        self._internal = None
        # v5.17.1 (audit P3-12): per-instance modal-basis LRU keyed on
        # (profile fingerprint, k0).  Dedups identical layers WITHIN a solve
        # (an ABAB... periodic stack pays one eig per DISTINCT profile, not
        # per repetition) and reuses eigs ACROSS set_source+solve sweep
        # calls -- the BOR analog of PMMStack.prepare()'s _eig_cache.
        # Byte-identical R/T: a cache hit returns the exact dict a recompute
        # would have produced (same assembled matrices -> same LAPACK bits).
        self._modal_cache = OrderedDict()

    # ---- geometry ------------------------------------------------------- #
    def add_layer(self, thickness, *, eps_profile=None, rings=None,
                  eps=None, eps_tensor_profile=None, eps_tensor=None,
                  segments=None):
        """Add a z-layer of given ``thickness``.

        Exactly one of: ``eps_profile`` (callable r-array->eps), ``rings``
        (``(period, duty, n_ridge, n_groove)`` concentric binary grating),
        ``eps`` (uniform scalar permittivity), ``eps_tensor_profile``
        (callable r-array -> ``(N, 3)`` array) or ``eps_tensor`` (a uniform
        3-tuple).

        ANISOTROPIC layers (``eps_tensor*``) carry the DIAGONAL CYLINDRICAL
        tensor ``diag(eps_rr, eps_phiphi, eps_zz)``.  That is exactly the class
        a body of revolution can represent: each component enters its own role
        in the radial operator -- ``eps_rr`` on the faces under the wall-normal
        inverse rule, ``eps_phiphi`` pointwise on the nodes, ``eps_zz`` in the
        ``E_z`` elimination.  An off-diagonal ``eps_r,phi`` (equivalently a
        uniform Cartesian director) is NOT azimuthally invariant, couples the
        ``m`` harmonics and is therefore rejected by the physics, not merely
        unimplemented -- a radially or azimuthally aligned uniaxial IS
        diagonal here and is exactly representable.  Anisotropic layers are
        NumPy-only (the JAX path raises).

        Layers with the SAME profile fingerprint (identical ``rings`` /
        ``eps`` parameters, or the same ``eps_profile`` callable object)
        share one eigensolve per ``k0`` via the modal LRU (audit P3-12).

        "Exactly one of" is now ENFORCED (audit W6-B7): passing two of them
        silently applied a precedence (``rings`` beat ``eps``, ``eps_profile``
        beat ``eps``) and quietly discarded the other."""
        _given = [nm for nm, v in (("eps_profile", eps_profile),
                                   ("rings", rings), ("eps", eps),
                                   ("eps_tensor_profile", eps_tensor_profile),
                                   ("eps_tensor", eps_tensor),
                                   ("segments", segments))
                  if v is not None]
        if len(_given) > 1:
            raise ValueError(
                "BORStack.add_layer: pass EXACTLY ONE of eps_profile / rings / "
                "eps / eps_tensor_profile / eps_tensor (got %s)."
                % (", ".join(_given),))
        if rings is not None:
            period, duty, n_r, n_g = rings
            # er/eg computed lazily (inside prof / the key) so a TRACED ring
            # index does not hit complex() at add_layer time (the NumPy prof /
            # key are unused on the JAX path).
            _rt = _is_jax_array(n_r) or _is_jax_array(n_g)

            def prof(r, period=period, duty=duty, nr=n_r, ng=n_g):
                return np.where((r % period) < duty * period,
                                complex(nr) ** 2, complex(ng) ** 2
                                ).astype(complex)
            fn = prof
            key = (("rings_t", float(period), float(duty), id(n_r), id(n_g))
                   if _rt else
                   ("rings", float(period), float(duty),
                    complex(n_r) ** 2, complex(n_g) ** 2))

            def jbuild(r_n, jnp, period=period, duty=duty, nr=n_r, ng=n_g):
                cj = jnp.complex128
                mask = jnp.asarray((r_n % period) < duty * period)
                return jnp.where(mask, jnp.asarray(nr).astype(cj) ** 2,
                                 jnp.asarray(ng).astype(cj) ** 2)
            raw = [n_r, n_g]
        elif eps_profile is not None:
            def prof(r, ep=eps_profile):
                return np.asarray(ep(r), dtype=complex)
            fn = prof

            def jbuild(r_n, jnp, ep=eps_profile):
                return jnp.asarray(ep(r_n)).astype(jnp.complex128)
            raw = []                               # traced-ness not auto-detected
            # Fingerprint by the USER callable's identity (the key tuple
            # keeps it alive, so the id can't be recycled); repeated
            # add_layer with the same callable dedups.  Unhashable callables
            # fall back to the per-call wrapper (unique -> never shared).
            try:
                hash(eps_profile)
                key = ("profile", eps_profile)
            except TypeError:
                key = ("profile", fn)
        elif eps is not None:
            def prof(r, e=eps):
                return np.full_like(r, complex(e), dtype=complex)
            fn = prof
            key = (("eps_t", id(eps)) if _is_jax_array(eps)
                   else ("eps", complex(eps)))

            def jbuild(r_n, jnp, e=eps):
                return jnp.full((r_n.shape[0],),
                                jnp.asarray(e).astype(jnp.complex128))
            raw = [eps]
        elif eps_tensor_profile is not None:
            def prof(r, ep=eps_tensor_profile):
                return np.asarray(ep(r), dtype=complex)
            fn = prof

            def jbuild(r_n, jnp, ep=eps_tensor_profile):
                raise NotImplementedError(
                    "BORStack: anisotropic layers have no JAX path yet -- use "
                    "an isotropic eps / eps_profile / rings layer for traced "
                    "solves.")
            raw = []
            try:
                hash(eps_tensor_profile)
                key = ("tprofile", eps_tensor_profile)
            except TypeError:
                key = ("tprofile", fn)
        elif eps_tensor is not None:
            if any(_is_jax_array(v) for v in np.asarray(eps_tensor,
                                                        dtype=object).ravel()):
                raise NotImplementedError(
                    "BORStack.add_layer: traced (JAX) eps_tensor components "
                    "are supported via segments=[(Rbig, (eps_rr, eps_phiphi, "
                    "eps_zz))] on basis='sem' -- a uniform anisotropic layer "
                    "is a single full-radius segment.")
            _t = np.asarray(eps_tensor, dtype=complex).ravel()
            if _t.size != 3:
                raise ValueError(
                    "BORStack.add_layer: eps_tensor must be the 3 DIAGONAL "
                    "CYLINDRICAL components (eps_rr, eps_phiphi, eps_zz); got "
                    "%d value(s)." % _t.size)

            def prof(r, t=_t):
                return np.repeat(t[None, :], np.asarray(r).size, axis=0)
            fn = prof

            def jbuild(r_n, jnp, t=_t):
                raise NotImplementedError(
                    "BORStack: anisotropic layers have no JAX path yet -- use "
                    "an isotropic eps / eps_profile / rings layer for traced "
                    "solves.")
            raw = []
            key = ("eps_tensor", complex(_t[0]), complex(_t[1]), complex(_t[2]))
        elif segments is not None:
            # explicit annuli: ((r_out_1, eps_1), ..., (Rbig, eps_K)) --
            # eps entries scalar or the diagonal cylindrical 3-tuple.  The
            # native spec of the SEM basis (walls are element boundaries);
            # also usable on the FD basis via the step profile below.
            segs = []
            r_prev = 0.0
            for r_out, ev in segments:
                r_out = float(r_out)
                if r_out <= r_prev:
                    raise ValueError(
                        "BORStack.add_layer(segments=...): r_out must be "
                        f"strictly increasing, got {r_out} after {r_prev}.")
                # each eps entry: scalar | (rr, pp, zz) triple | and any
                # component may be a DISPERSIVE wl->eps callable (resolved at
                # solve time on the SEM basis; the FD step profile below only
                # supports concrete values)
                if callable(ev):
                    tri = (ev,) * 3
                elif _is_jax_array(ev):
                    # traced (or concrete-jnp) eps stays a RAW jax scalar so
                    # gradients flow on the SEM twin; checked BEFORE np.asarray
                    # (which would silently concretize a concrete jnp array)
                    _shp = tuple(getattr(ev, "shape", ()))
                    if _shp in ((), (1,)):
                        tri = ((ev if _shp == () else ev[0]),) * 3
                    elif _shp == (3,):
                        tri = (ev[0], ev[1], ev[2])
                    else:
                        raise ValueError(
                            "BORStack.add_layer(segments=...): a JAX eps must "
                            "be a scalar or a length-3 (eps_rr, eps_phiphi, "
                            f"eps_zz) vector; got shape {_shp}.")
                else:
                    # tuples/lists are taken per-component WITHOUT np.asarray
                    # (which probes element __array__ even at dtype=object, so
                    # a traced component would raise -- and a CONCRETE jnp
                    # component would silently concretize off the traced path)
                    if isinstance(ev, (tuple, list)):
                        a = list(ev)
                    else:
                        a = list(np.asarray(ev, dtype=complex).ravel())
                    if len(a) not in (1, 3):
                        raise ValueError(
                            "BORStack.add_layer(segments=...): each eps must "
                            "be a scalar or a (eps_rr, eps_phiphi, eps_zz) "
                            f"triple; got {len(a)} value(s).")
                    def _coerce(v):
                        return (v if callable(v) or _is_jax_array(v)
                                else complex(v))
                    tri = ((_coerce(a[0]),) * 3 if len(a) == 1 else
                           (_coerce(a[0]), _coerce(a[1]), _coerce(a[2])))
                segs.append((r_out, tri))
                r_prev = r_out
            if abs(r_prev - self.Rbig) > 1e-9 * self.Rbig:
                raise ValueError(
                    "BORStack.add_layer(segments=...): the last r_out must "
                    f"equal Rbig={self.Rbig!r}, got {r_prev!r}.")
            _bnd = np.asarray([0.0] + [rs for rs, _t in segs])
            _disp = any(callable(v) for _rs, t in segs for v in t)
            _traced = any(_is_jax_array(v) for _rs, t in segs for v in t)
            if _disp or _traced:
                def prof(r):
                    raise NotImplementedError(
                        "BORStack: dispersive (callable) or traced (JAX) "
                        "segment eps is supported on basis='sem' only; the "
                        "FD step profile needs concrete values.")
            else:
                _tri = np.asarray([t for _rs, t in segs], dtype=complex)
                _iso = bool(np.all(_tri[:, 0] == _tri[:, 1])
                            and np.all(_tri[:, 1] == _tri[:, 2]))

                def prof(r, b=_bnd, t=_tri, iso=_iso):
                    idx = np.clip(np.searchsorted(b, np.asarray(r, float),
                                                  side="right") - 1,
                                  0, t.shape[0] - 1)
                    return t[idx, 0] if iso else t[idx, :]
            fn = prof
            # traced entries are unhashable -> id-keyed (the traced path never
            # consults the modal LRU; the key only has to be distinct)
            key = ("segments", tuple(
                (rs, tuple(id(v) if _is_jax_array(v) else v for v in t))
                for rs, t in segs))

            def jbuild(r_n, jnp):
                raise NotImplementedError(
                    "BORStack: segments= layers have no FD JAX path -- use "
                    "basis='sem' for traced segment solves.")
            raw = [v for _rs, t in segs for v in t if _is_jax_array(v)]
        else:
            raise ValueError(
                "add_layer needs eps_profile, rings, eps, eps_tensor_profile, "
                "eps_tensor or segments")
        # Thickness validation (audit P3-10) -- mirror PMM2DStack.add_layer:
        # a NEGATIVE thickness flips the propagation exponent exp(iqL) so
        # forward-oriented evanescent modes GROW, silently destabilizing the
        # Redheffer cascade instead of raising.
        if _is_jax_array(thickness):
            thk_store = thickness                 # traced thickness stays raw
        else:
            thickness = float(thickness)
            if not np.isfinite(thickness) or thickness <= 0.0:
                raise ValueError("BORStack.add_layer: thickness must be > 0")
            thk_store = thickness
        # SEM sidecar: (interior walls, full segment list) when the spec
        # carries wall information; None otherwise (SEM solve rejects it).
        if segments is not None:
            _walls = tuple(rs for rs, _t in segs if rs < self.Rbig)
            self._sem_layers.append((_walls, tuple(segs)))
        elif eps is not None:
            # traced eps stays RAW in the sidecar (the SEM twin differentiates
            # through it); concrete eps is pinned to a hashable complex
            _e = eps if _is_jax_array(eps) else complex(eps)
            self._sem_layers.append(((), ((self.Rbig, (_e,) * 3),)))
        elif eps_tensor is not None:
            self._sem_layers.append(
                ((), ((self.Rbig, (complex(_t[0]), complex(_t[1]),
                                   complex(_t[2]))),)))
        elif rings is not None and not _rt:
            _per, _duty, _nr, _ng = (float(rings[0]), float(rings[1]),
                                     complex(rings[2]), complex(rings[3]))
            _er, _eg = _nr ** 2, _ng ** 2
            _sem_segs, _edge = [], 0.0
            while _edge < self.Rbig - 1e-9 * self.Rbig:
                _ridge_hi = min(_edge + _duty * _per, self.Rbig)
                _sem_segs.append((_ridge_hi, (_er,) * 3))
                if _ridge_hi >= self.Rbig - 1e-9 * self.Rbig:
                    break
                _groove_hi = min(_edge + _per, self.Rbig)
                _sem_segs.append((_groove_hi, (_eg,) * 3))
                _edge += _per
            _sem_segs[-1] = (self.Rbig, _sem_segs[-1][1])
            self._sem_layers.append(
                (tuple(rs for rs, _t in _sem_segs[:-1]),
                 tuple(_sem_segs)))
        else:
            self._sem_layers.append(None)
        self._layers.append((thk_store, fn, key))
        self._jax_layers.append((thk_store, jbuild, raw))
        self._last = None       # geometry change supersedes retained state
        self._internal = None
        return self

    def _is_jax(self):
        """True if any traced layer eps / ring index / thickness is a JAX array
        (the half-spaces stay concrete)."""
        return any(_is_jax_array(v)
                   for thk, _jb, raw in self._jax_layers
                   for v in ([thk] + list(raw)))

    def set_source(self, wavelength=None, *, k0=None):
        # Source validation (audit P3-10): wavelength <= 0 gives k0 = inf /
        # negative, and solve() then silently returns EMPTY R/T.
        # W6-B7: give BOTH and the wavelength used to be silently discarded.
        if wavelength is not None and k0 is not None:
            raise ValueError(
                "BORStack.set_source: give wavelength OR k0, not both (got "
                f"wavelength={wavelength!r}, k0={k0!r}).")
        if k0 is None:
            if wavelength is None:
                raise ValueError(
                    "BORStack.set_source: give wavelength or k0.")
            wavelength = float(wavelength)
            if not np.isfinite(wavelength) or wavelength <= 0.0:
                raise ValueError(
                    f"BORStack.set_source: wavelength must be > 0, got "
                    f"{wavelength}.")
            self.k0 = 2 * np.pi / wavelength
        else:
            k0 = float(k0)
            if not np.isfinite(k0) or k0 <= 0.0:
                raise ValueError(
                    f"BORStack.set_source: k0 must be > 0, got {k0}.")
            self.k0 = k0
        self._last = None       # source change supersedes retained state
        self._internal = None
        return self

    # ---- solve ---------------------------------------------------------- #
    def _build(self, profile_key, eps_fn):
        """Modal basis of one layer, memoized on (profile fingerprint, k0)
        in the per-instance LRU (audit P3-12).  Identical profiles share one
        dense eigensolve within a solve() AND across set_source+solve sweep
        calls; a hit returns the SAME dict a recompute would (solve() never
        mutates it)."""
        ck = (profile_key, self.k0, self.m, self.Rbig, self.N)
        hit = self._modal_cache.get(ck)
        if hit is not None:
            self._modal_cache.move_to_end(ck)     # LRU touch
            return hit
        L = layer_modes(self.m, self.Rbig, self.N, eps_fn, self.k0,
                        staggered=True)
        self._modal_cache[ck] = L
        if len(self._modal_cache) > _MODAL_CACHE_SIZE:
            self._modal_cache.popitem(last=False)
        return L

    def solve(self, *, retain_internal=False):
        """Cascade the stack and return per-propagating-order R/T efficiencies.

        See the class docstring's "Terminology (mode vs order)" note for the
        precise ``m`` (azimuthal order) / order (diffraction channel) / mode
        (modal-basis column) distinction used below.

        Returns a dict (the current pre-S5-6 result container -- see the class
        docstring's "Result container" note): ``q`` (incident propagating
        axial wavenumbers, one per order), ``gamma`` (their transverse
        wavenumbers), ``angles`` (incident-order polar angle in the
        SUPERSTRATE, rad -- computed from ``eps_sup`` and the superstrate
        order ``q``; for ``n_substrate != n_superstrate`` the transmitted
        orders' substrate-side angles differ by Snell refraction, audit
        P3-11), ``R``/``T`` (reflected/transmitted power fraction summed over
        propagating orders), ``energy`` (R+T per incident order), ``S`` (the
        global S-matrix in the modal basis).

        MODAL GAUGE of ``S`` (AUDIT_DYNAMETA_CONSUMER_API_GAPS C1a): the
        columns are FLUX-normalized (every propagating mode carries unit
        z-flux, forward-oriented), so ``|S11[jp, j]|^2`` is a true power
        fraction -- but each mode profile carries the LAPACK-arbitrary
        eigenvector PHASE.  DIAGONAL entries ``S11[j, j]`` are
        gauge-INVARIANT (incident and reflected mode share one column:
        backward = ``[W; -V]``), so the fundamental-mode reflection PHASE may
        be read off directly; off-diagonal phases are basis-relative.  Use
        :meth:`per_mode_amplitudes` for amplitudes in a PINNED deterministic
        gauge.

        ``retain_internal=True`` additionally retains the per-layer partial
        cascades for :meth:`layer_absorption` (NumPy path only).

        The staggered basis is spurious-free, so NO reldiv filter is needed (the
        propagating criterion alone is exact) -- which lets us skip the two
        extra half-space eigensolves the reldiv tags would have cost.  Layers
        with identical profile fingerprints (incl. super/substrate) share one
        eig via the modal LRU (audit P3-12)."""
        if self.k0 is None:
            raise RuntimeError("call set_source(...) before solve()")
        if self.basis == "sem":
            return self._solve_sem(retain_internal=retain_internal)
        # every solve supersedes the retained state (audit-P1-04 contract)
        self._last = None
        self._internal = None
        if self._is_jax():
            if retain_internal:
                raise NotImplementedError(
                    "BORStack.solve(retain_internal=True): not available on "
                    "the JAX (differentiable) path; use NumPy inputs for "
                    "layer_absorption.")
            # Differentiable twin: gradients flow through the traced layer eps /
            # ring index / thickness (half-spaces concrete).  Returns R/T as
            # full-2N masked arrays -- sum(R)/sum(T) match this NumPy path.
            from ._jax_bor import _jax_bor_stack_solve
            return _jax_bor_stack_solve(self)
        k0 = self.k0
        sup = self._build(("eps", self.eps_sup),
                          lambda r: np.full_like(r, self.eps_sup, dtype=complex))
        sub = (sup if self.eps_sub == self.eps_sup
               else self._build(("eps", self.eps_sub),
                                lambda r: np.full_like(r, self.eps_sub,
                                                       dtype=complex)))
        # Per-solve memo ON TOP of the LRU (audit W6-B8).  The LRU alone makes
        # the documented "an ABAB... periodic stack pays one eig per DISTINCT
        # profile, not per repetition" promise FALSE as soon as the number of
        # distinct profiles exceeds _MODAL_CACHE_SIZE: a cyclic sweep is the
        # LRU worst case, so every entry is evicted exactly before it is needed
        # again and the hit rate collapses to ZERO (measured: 20 distinct
        # profiles x 2 repetitions = 41 eigs instead of 21, and 41 again on
        # every re-solve).  This local dict makes the WITHIN-solve dedup
        # cap-independent; the LRU still governs ACROSS-solve reuse.  It holds
        # only references the ``mids`` list already holds, so no extra memory.
        _seen = {}
        mids = []
        for thk, fn, key in self._layers:
            L = _seen.get(key)
            if L is None:
                L = _seen[key] = self._build(key, fn)
            mids.append((thk, L))
        # cascade: sup -> [mid prop mid] ... -> sub.  The interface list is
        # precomputed (same matrices the inline build produced -- bit-identical
        # cascade) so retain_internal can reuse it for the partial cascades.
        nlay = len(mids)
        if mids:
            ifc = [interface_smatrix(sup["W"], sup["V"],
                                     mids[0][1]["W"], mids[0][1]["V"])]
            for i in range(1, nlay):
                ifc.append(interface_smatrix(
                    mids[i - 1][1]["W"], mids[i - 1][1]["V"],
                    mids[i][1]["W"], mids[i][1]["V"]))
            ifc.append(interface_smatrix(mids[-1][1]["W"], mids[-1][1]["V"],
                                         sub["W"], sub["V"]))
            S = ifc[0]
            for i, (thk, L) in enumerate(mids):
                S = redheffer_star(S, propagation_smatrix(L["q"], thk))
                S = redheffer_star(S, ifc[i + 1])
        else:
            S = interface_smatrix(sup["W"], sup["V"], sub["W"], sub["V"])
        if retain_internal and mids:
            # Partial cascades bracketing each layer (the RCWAStack /
            # PMMStack _internal_partials pattern): S_above[i] = sup -> TOP
            # of layer i (through the interface INTO it, before its own
            # propagation); S_below_bot[i] = BOTTOM of layer i -> sub
            # (without its propagation), so backward amplitudes reference
            # the layer BOTTOM and both exponentials DECAY.
            S_above = [None] * nlay
            S_above[0] = ifc[0]
            for i in range(1, nlay):
                S_above[i] = redheffer_star(
                    redheffer_star(S_above[i - 1], propagation_smatrix(
                        mids[i - 1][1]["q"], mids[i - 1][0])), ifc[i])
            S_below_bot = [None] * nlay
            S_below_bot[nlay - 1] = ifc[nlay]
            for i in range(nlay - 2, -1, -1):
                S_below_bot[i] = redheffer_star(
                    redheffer_star(ifc[i + 1], propagation_smatrix(
                        mids[i + 1][1]["q"], mids[i + 1][0])),
                    S_below_bot[i + 1])
            self._internal = dict(S_above=S_above, S_below_bot=S_below_bot)
        S11, S12, S21, S22 = S
        q = sup["q"]

        def prop(L, eps):
            # Classify in the DIMENSIONLESS axial index q/k0 (audit P2-06):
            # absolute thresholds on q (units 1/length) silently returned
            # empty R/T for small-k0 unit systems (e.g. nm-scale).
            #
            # AUDIT_BOR_PROPAGATING_CUTOFF_ENERGY_2026_07_13: the original
            # P2-06 constant (0.05, chosen to keep k0=2.0 bit-identity with
            # the pre-fix absolute threshold) was an ANGULAR cutoff -- it
            # dropped genuinely propagating near-grazing orders (theta up to
            # 88 deg in n=1.41), silently biasing per-order R/T low and
            # leaking energy (2.28e-2 on the ring-grating reproducer).  A
            # propagating mode is real-q up to the q ~ 0 degenerate point,
            # so the real-axis floor guards ONLY that point (1e-6).  This
            # floor is compatible with the flux normalizer's fallback branch
            # (zcascade: field-norm when |P| <= 1e-10 * fnrm): the modal
            # flux ratio scales as P/fnrm = qn for the limiting polarization
            # family (verified empirically), so every kept mode sits >= 4
            # decades above the fallback -- kept implies flux-normalized,
            # and |S|^2 stays a true power fraction.
            #
            # S1-16 (audit AUDIT_V5_24_2): this shares the {imag, real-floor,
            # index-ceiling} core with the twins bor_solve._physical_propagating
            # and _jax_bor._mask.  It carries NO reldiv leg BY DESIGN: BORStack
            # is staggered-only (div-conforming, spurious-free), so reldiv is
            # structurally 0 and the reldiv eigensolve is deliberately skipped
            # (see the solve() docstring).  Only bor_solve's optional NODAL
            # basis needs the reldiv leg to reject its spurious sea.
            qn = L["q"] / k0
            return np.where((np.abs(qn.imag) < 5e-5) & (qn.real > 1e-6)
                            & (np.sqrt(eps).real - qn.real > -5e-10))[0]
        inc = prop(sup, self.eps_sup)
        out = prop(sub, self.eps_sub)
        R = np.array([np.sum([abs(S11[jp, j]) ** 2 for jp in inc]) for j in inc])
        T = np.array([np.sum([abs(S21[jp, j]) ** 2 for jp in out]) for j in inc])
        qi = q[inc].real
        gamma = np.sqrt(np.maximum(self.eps_sup.real * k0 ** 2 - qi ** 2, 0.0))
        angles = np.arcsin(np.clip(gamma / (np.sqrt(self.eps_sup.real) * k0),
                                   0, 1))
        # retained state for per_mode_amplitudes / layer_absorption (cheap:
        # references into the modal LRU + the S-matrix already returned)
        self._last = dict(S=S, sup=sup, sub=sub, mids=mids, inc=inc, out=out,
                          k0=k0, R=R, T=T)
        return dict(q=qi, gamma=gamma, angles=angles, R=R, T=T, energy=R + T,
                    inc=inc, out=out, S=S)

    def _solve_sem(self, *, retain_internal=False):
        """SEM-basis solve: per-layer ring-wall-aligned element meshes
        (window +-1 neighbour enrichment, the Cartesian per-layer-grids
        pattern), cross-tested Galerkin mortar interfaces, same cascade and
        R/T extraction as the FD path.  See sem_radial's module docstring."""
        from .sem_radial import (
            SemRadialMesh,
            equalize_meshes,
            sem_interface_smatrix,
            sem_layer_modes,
        )
        if self._is_jax():
            if retain_internal:
                raise NotImplementedError(
                    "BORStack.solve(retain_internal=True): not available on "
                    "the JAX (differentiable) path; use NumPy inputs for "
                    "layer_absorption.")
            # Differentiable SEM twin: gradients flow through traced segment /
            # uniform-layer eps (scalar or the diagonal cylindrical triple)
            # and layer thicknesses; walls, meshes and half-spaces stay
            # concrete.  Masked full-array R/T contract (see _jax_sem).
            from ._jax_sem import _jax_sem_stack_solve
            return _jax_sem_stack_solve(self)
        self._last = None
        self._internal = None
        k0 = self.k0
        wl_now = 2.0 * np.pi / k0
        sem_layers = []
        for i, sc in enumerate(self._sem_layers):
            if sc is None:
                raise ValueError(
                    f"BORStack(basis='sem'): layer {i} was added with a bare "
                    "profile callable, which carries no ring-wall positions "
                    "to mesh on -- re-add it with segments=[(r_out, eps), "
                    "...] (or rings= / eps= / eps_tensor=).")
            walls_i, segs_i = sc
            resolved = []
            for r_out, tri in segs_i:
                resolved.append((r_out, tuple(
                    complex(t(wl_now)) if callable(t) else complex(t)
                    for t in tri)))
            sem_layers.append((walls_i, tuple(resolved)))
        # window +-1 wall enrichment (half-spaces are wall-free neighbours)
        walls = [sc[0] for sc in sem_layers]
        win = []
        for i in range(len(walls)):
            u = set(walls[i])
            if i > 0:
                u |= set(walls[i - 1])
            if i + 1 < len(walls):
                u |= set(walls[i + 1])
            win.append(sorted(u))
        sup_walls = sorted(set(walls[0])) if walls else []
        sub_walls = sorted(set(walls[-1])) if walls else []

        def _refine(bnd):
            # hp knob: split every interval into elements_per_segment
            # sub-elements, Chebyshev-Lobatto graded toward BOTH ends (the
            # Cartesian PMM _graded_boundaries recipe -- resolves the
            # metal-corner field concentration at ring walls)
            k = self.elements_per_segment
            if k == 1:
                return bnd
            if self.grade:
                sloc = 0.5 * (1.0 - np.cos(np.pi * np.arange(k + 1) / k))
            else:
                sloc = np.linspace(0.0, 1.0, k + 1)
            out = [bnd[0]]
            for lo, hi in zip(bnd[:-1], bnd[1:]):
                out.extend(lo + (hi - lo) * sloc[1:])
            return np.asarray(out)

        def build_mesh(wall_list, segs):
            bnd = np.concatenate([[0.0], np.asarray(wall_list, float),
                                  [self.Rbig]])
            bnd = np.unique(bnd)
            keep = np.concatenate([[True], np.diff(bnd) > 1e-12 * self.Rbig])
            bnd = bnd[keep]
            if abs(bnd[-1] - self.Rbig) > 0:
                bnd[-1] = self.Rbig
            bnd = _refine(bnd)
            # WAVELENGTH-RESOLUTION CAP (the SEM analog of the FD grid's N;
            # found the hard way: a wall-sparse layer left a ~7-wavelength
            # element at degree 8, and the starved modal basis broke the
            # cascade -- NEGATIVE absorption in a lossless stack).  Split any
            # element wider than degree * lambda_local / DPW so every element
            # carries >= DPW dofs per local wavelength in its own medium.
            DPW = 8.0
            edges = np.asarray([rs for rs, _t in segs])
            tris = [t for _rs, t in segs]
            out = [float(bnd[0])]
            for lo, hi in zip(bnd[:-1], bnd[1:]):
                jdx = int(np.searchsorted(edges, 0.5 * (lo + hi),
                                          side="left"))
                tri = tris[min(jdx, len(tris) - 1)]
                n_loc = max(abs(np.sqrt(complex(t)).real) for t in tri)
                n_loc = max(n_loc, 1e-3)
                lam_loc = 2.0 * np.pi / (k0 * n_loc)
                max_el = self.degree * lam_loc / DPW
                nsplit = max(1, int(np.ceil((hi - lo) / max_el)))
                out.extend(np.linspace(lo, hi, nsplit + 1)[1:])
            bnd = np.asarray(out)
            eps_el = []
            edges = np.asarray([rs for rs, _t in segs])
            tris = [t for _rs, t in segs]
            for lo, hi in zip(bnd[:-1], bnd[1:]):
                jdx = int(np.searchsorted(edges, 0.5 * (lo + hi),
                                          side="left"))
                eps_el.append(tris[min(jdx, len(tris) - 1)])
            return SemRadialMesh(bnd, eps_el, self.degree)

        sup_segs = ((self.Rbig, (self.eps_sup,) * 3),)
        sub_segs = ((self.Rbig, (self.eps_sub,) * 3),)
        meshes = ([build_mesh(sup_walls, sup_segs)]
                  + [build_mesh(win[i], sem_layers[i][1])
                     for i in range(len(walls))]
                  + [build_mesh(sub_walls, sub_segs)])
        meshes = equalize_meshes(meshes)

        # per-solve memo on top of the LRU (the audit-W6-B8 pattern: a
        # stack with more distinct meshes than _MODAL_CACHE_SIZE would
        # otherwise thrash the LRU cyclically across repeated solves)
        _seen = {}

        def modes(mesh):
            ck = ("sem", mesh.fingerprint(), k0, self.m, self.degree)
            L = _seen.get(ck)
            if L is not None:
                return L
            L = self._modal_cache.get(ck)
            if L is not None:
                self._modal_cache.move_to_end(ck)    # LRU touch
                _seen[ck] = L
                return L
            L = sem_layer_modes(mesh, self.m, k0)
            _seen[ck] = L
            self._modal_cache[ck] = L
            if len(self._modal_cache) > _MODAL_CACHE_SIZE:
                self._modal_cache.popitem(last=False)
            return L

        sup = modes(meshes[0])
        sub = modes(meshes[-1])
        mids = [(thk, modes(meshes[1 + i]))
                for i, (thk, _fn, _key) in enumerate(self._layers)]
        nlay = len(mids)
        if mids:
            ifc = [sem_interface_smatrix(sup, mids[0][1])]
            for i in range(1, nlay):
                ifc.append(sem_interface_smatrix(mids[i - 1][1], mids[i][1]))
            ifc.append(sem_interface_smatrix(mids[-1][1], sub))
            S = ifc[0]
            for i, (thk, L) in enumerate(mids):
                S = redheffer_star(S, propagation_smatrix(L["q"], thk))
                S = redheffer_star(S, ifc[i + 1])
        else:
            S = sem_interface_smatrix(sup, sub)
        if retain_internal and mids:
            # partial cascades bracketing each layer (the FD solve() /
            # PMMStack _internal_partials pattern, verbatim semantics)
            S_above = [None] * nlay
            S_above[0] = ifc[0]
            for i in range(1, nlay):
                S_above[i] = redheffer_star(
                    redheffer_star(S_above[i - 1], propagation_smatrix(
                        mids[i - 1][1]["q"], mids[i - 1][0])), ifc[i])
            S_below_bot = [None] * nlay
            S_below_bot[nlay - 1] = ifc[nlay]
            for i in range(nlay - 2, -1, -1):
                S_below_bot[i] = redheffer_star(
                    redheffer_star(ifc[i + 1], propagation_smatrix(
                        mids[i + 1][1]["q"], mids[i + 1][0])),
                    S_below_bot[i + 1])
            self._internal = dict(S_above=S_above, S_below_bot=S_below_bot)
        S11, S12, S21, S22 = S
        q = sup["q"]

        def prop(L, eps):
            # same dimensionless propagating gate as the FD path (P2-06 +
            # AUDIT_BOR_PROPAGATING_CUTOFF_ENERGY_2026_07_13)
            qn = L["q"] / k0
            return np.where((np.abs(qn.imag) < 5e-5) & (qn.real > 1e-6)
                            & (np.sqrt(eps).real - qn.real > -5e-10))[0]

        inc = prop(sup, self.eps_sup)
        out = prop(sub, self.eps_sub)
        R = np.array([np.sum([abs(S11[jp, j]) ** 2 for jp in inc])
                      for j in inc])
        T = np.array([np.sum([abs(S21[jp, j]) ** 2 for jp in out])
                      for j in inc])
        qi = q[inc].real
        gamma = np.sqrt(np.maximum(self.eps_sup.real * k0 ** 2 - qi ** 2,
                                   0.0))
        angles = np.arcsin(np.clip(gamma / (np.sqrt(self.eps_sup.real) * k0),
                                   0, 1))
        self._last = dict(S=S, sup=sup, sub=sub, mids=mids, inc=inc, out=out,
                          k0=k0, R=R, T=T)
        return dict(q=qi, gamma=gamma, angles=angles, R=R, T=T,
                    energy=R + T, inc=inc, out=out, S=S)

    # ---- retained-state observables (AUDIT_DYNAMETA_CONSUMER_API_GAPS C1) -- #

    def per_mode_amplitudes(self, port="reflection"):
        """Complex modal scattering amplitudes of the last :meth:`solve` in a
        PINNED deterministic gauge, restricted to the propagating channels.

        Returns a dict: ``amplitude`` -- ``(n_out, n_inc)`` complex matrix
        (column j = response to unit-flux incident propagating superstrate
        mode ``q_inc[j]``; row p = outgoing propagating mode ``q_out[p]``,
        reflected into the superstrate or transmitted into the substrate per
        ``port``); ``q_inc`` / ``q_out`` -- the axial wavenumbers of those
        channels; ``k0``.  ``abs(amplitude)**2`` sums to the solve's ``R``/
        ``T`` per incident mode (flux-normalized modes).

        GAUGE: raw eigenvector columns carry a LAPACK-arbitrary phase, so raw
        ``S11[jp, j]`` off-diagonal phases are basis-relative (the DIAGONAL is
        gauge-invariant -- incident and reflected share one column).  Here
        every mode column is pinned to the deterministic convention 'the
        dominant field sample of the mode profile is real-positive'
        (``A = ph_out * S * conj(ph_in)``), making all entries reproducible,
        physically-phased amplitudes; the diagonal is unchanged by
        construction."""
        if port not in ("reflection", "transmission"):
            raise ValueError(
                f"BORStack.per_mode_amplitudes: port must be 'reflection' or "
                f"'transmission', got {port!r}.")
        d = self._last
        if d is None:
            raise ValueError(
                "BORStack.per_mode_amplitudes: no retained solve -- run a "
                "NumPy solve() first (any add_layer / set_source / re-solve "
                "supersedes it; the JAX path retains nothing).")
        S11, _S12, S21, _S22 = d["S"]
        inc, out = d["inc"], d["out"]
        ph_in = _column_phases(d["sup"]["W"])[inc]
        if port == "reflection":
            Sblk, ph_out = S11, ph_in
            q_out = d["sup"]["q"][inc]
        else:
            Sblk = S21
            ph_out = _column_phases(d["sub"]["W"])[out]
            q_out = d["sub"]["q"][out]
        rows = inc if port == "reflection" else out
        A = (ph_out[:, None] * Sblk[np.ix_(rows, inc)]) * np.conj(ph_in)[None, :]
        return dict(amplitude=A, q_inc=d["sup"]["q"][inc].copy(),
                    q_out=np.asarray(q_out).copy(), k0=d["k0"])

    def layer_absorption(self):
        """Per-layer absorbed power fraction of the last
        ``solve(retain_internal=True)`` -- ``(n_layers, n_inc)`` (one column
        per incident propagating superstrate mode, unit incident flux), via
        the z-flux DIFFERENCE across each layer: the total staggered-basis
        Poynting flux (two-grid quadrature, audit P3-14) evaluated at the
        layer's top and bottom planes from the recovered forward/backward
        modal amplitudes.  ``R + T + sum_i A_i = 1`` per incident mode
        (machine precision; identically 0 per layer for lossless stacks) --
        the ``PMMStack.layer_absorption`` recipe in cylindrical coordinates
        (AUDIT_DYNAMETA_CONSUMER_API_GAPS C1b)."""
        d = self._last
        if d is None or self._internal is None:
            raise ValueError(
                "BORStack.layer_absorption: the MOST RECENT solve must be "
                "solve(retain_internal=True) on NumPy inputs (any re-solve / "
                "add_layer / set_source supersedes the retained state).")
        mids = d["mids"]
        inc = d["inc"]
        sup = d["sup"]
        S_above = self._internal["S_above"]
        S_below_bot = self._internal["S_below_bot"]
        n2 = sup["W"].shape[0]          # 2N
        # unit-flux incident columns: identity restricted to the inc set
        cinc = np.zeros((n2, len(inc)), dtype=complex)
        cinc[inc, np.arange(len(inc))] = 1.0
        is_sem = sup.get("mesh") is not None      # SEM dicts carry the mesh
        A = np.zeros((len(mids), len(inc)))
        eye = np.eye(n2, dtype=complex)
        for i, (thk, L) in enumerate(mids):
            Sa = S_above[i]
            Sb_bot = S_below_bot[i]
            # S_below[i] = prop(i) * S_below_bot[i]; only its 11 block enters
            Xf = np.exp(1j * L["q"] * thk)
            Sb11 = Xf[:, None] * Sb_bot[0] * Xf[None, :]
            c_fwd = np.linalg.solve(eye - Sa[3] @ Sb11, Sa[2] @ cinc)
            c_bwd = Sb_bot[0] @ (Xf[:, None] * c_fwd)     # at layer BOTTOM
            W, V, qL = L["W"], L["V"], L["q"]
            Xd = np.exp(1j * qL * thk)
            # top (z = 0): fwd referenced here; bwd decays up from the bottom
            E_top = W @ c_fwd + W @ (Xd[:, None] * c_bwd)
            H_top = V @ c_fwd - V @ (Xd[:, None] * c_bwd)
            # bottom (z = thk)
            E_bot = W @ (Xd[:, None] * c_fwd) + W @ c_bwd
            H_bot = V @ (Xd[:, None] * c_fwd) - V @ c_bwd
            # per-layer weights: FD layers share one grid (sup's weights
            # are everyone's); SEM layers each carry their own mesh
            wq_f = np.real(L["wq_face"])
            wq_n = np.real(L["wq_node"])
            if is_sem:
                A[i] = (_sem_flux(E_top, H_top, wq_f, wq_n, L["n1"])
                        - _sem_flux(E_bot, H_bot, wq_f, wq_n, L["n1"]))
            else:
                A[i] = (_stag_flux(E_top, H_top, wq_f, wq_n)
                        - _stag_flux(E_bot, H_bot, wq_f, wq_n))
        return A
