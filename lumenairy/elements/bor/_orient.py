"""THE ONE forward-orientation kernel of the BOR (body-of-revolution) engines,
and the two companion decisions that travel with it.

WHAT THIS MODULE IS FOR.  An axisymmetric layer's modal solve returns a set of
SQUARED axial wavenumbers ``q^2``; taking the square root leaves a sign
ambiguity that the S-matrix cascade cannot tolerate, because ``+q`` and ``-q``
are the FORWARD (``+z``) and BACKWARD mode of the same field pattern and the
cascade's whole bookkeeping is which set is which.  The BOR engines resolve it
by physics rather than by a sign test on a rounding-level quantity:

* a PROPAGATING mode is oriented by the sign of its own ``r dr``-weighted axial
  Poynting flux -- a forward mode carries power in ``+z``;
* an EVANESCENT mode is oriented by decay in ``+z`` (``Im q > 0``), so the
  propagator ``exp(i q L)`` shrinks over a forward thickness and the cascade is
  unconditionally stable.

THE FIVE COPIES this module replaces (5.45.1):

    lumenairy/elements/bor/zcascade.py:86      staggered FD (basis='fd'), per-mode loop
    lumenairy/elements/bor/zcascade.py:227     legacy nodal FD, per-mode loop
    lumenairy/elements/bor/sem_radial.py:428   SEM (basis='sem'), vectorized
    lumenairy/elements/bor/_jax_bor.py:98      FD JAX twin, vectorized jnp
    lumenairy/elements/bor/_jax_sem.py:247     SEM JAX twin, vectorized jnp

Five bodies of one decision is the shape that bred the six-copy factor-``i``
defect (audit S1-8) and the six-copy branch-cut defect
(``FIX_BRANCH_CUT_ROUND2_2026_09_11.md``): a fix lands in the copy the author
was looking at and the other four keep the defect, silently, because nothing
imports them.  The decision is that there is exactly ONE, ``xp=``-parametrized
so the eager NumPy paths and the traced ``jnp`` twins run the SAME object
rather than two bodies that are believed to agree.
``tests/unit/test_fix_bor_multilayer_guards.py`` pins that with a grep over the
package.

THE CLASSIFIER BAND IS A SEPARATE QUESTION FROM THE NUMBER OF COPIES.  The
module was introduced carrying the SHIPPED band unchanged, so the consolidation
could be proved to move nothing (148 BOR gates plus 30 hashed R/T fixtures,
bit-identical on both builds); the band then moved in its own commit, to the
shape every other engine in the library uses.  What was wrong with the shipped
band, and the two-sided measurement that replaced it, is in
:func:`orient_band_scale`.
"""
from __future__ import annotations

from ...backend.array import array_namespace

#: THE CLASSIFIER BAND.  ``|Im q| <= band * scale`` calls a mode PROPAGATING
#: and hands its orientation to the flux; outside the band the mode is
#: EVANESCENT and is oriented by decay.  1e-8 is the same factor
#: ``rcwa/_core._CUT_BAND_REL`` and ``pmm/_core._forward_branch_flip`` carry;
#: the SCALE it multiplies is what 5.45.1 fixed -- see
#: :func:`orient_band_scale` for the populations and their margins.
_BOR_CUT_BAND_REL = 1e-8

#: The flux-normalizer's FALLBACK threshold: a mode whose ``|z-flux|`` is below
#: this fraction of its own ``r dr`` field norm is normalized by the field norm
#: instead (there is no meaningful flux to normalize to).  RELATIVE to the
#: mode's own norm on purpose -- both sides scale as ``field^2 * length^2``, so
#: the ratio is unit-invariant, where the absolute threshold it replaced
#: silently mis-normalized meter-scale inputs (audit P1-01).
_BOR_FLUX_FALLBACK_REL = 1e-10

#: The R/T CHANNEL GATE's shared core, in the DIMENSIONLESS axial index
#: ``qn = q / k0`` (audit P2-06: absolute thresholds on ``q``, which carries
#: units of inverse length, silently returned an empty R/T set for small-``k0``
#: unit systems).  ``_BOR_CHANNEL_IMAG_BAR`` admits a mode as a real channel;
#: ``_BOR_CHANNEL_REAL_FLOOR`` guards ONLY the ``q ~ 0`` degenerate point --
#: it is deliberately NOT an angular cutoff, which is what
#: ``AUDIT_BOR_PROPAGATING_CUTOFF_ENERGY_2026_07_13`` found dropping genuinely
#: propagating near-grazing orders and leaking 2.28e-2 of the energy.
_BOR_CHANNEL_IMAG_BAR = 5e-5
_BOR_CHANNEL_REAL_FLOOR = 1e-6

def orient_band_scale(q, k0, *, xp=None):
    """The scale the propagating/evanescent classifier band is taken RELATIVE
    to, for a layer's spectrum ``q`` at vacuum wavenumber ``k0``:
    ``max(max|q| over the layer's spectrum, k0)``.

    WHAT THE FIVE COPIES DID UNTIL 5.45.1, AND WHY IT WAS A DEFECT.  They
    scaled the band by **the mode's OWN** ``|Re q|``, floored at 1e-300::

        prop = |Im q| < 1e-9 * max(|Re q|, 1e-300)

    Every other band of this shape in the library scales by the SPECTRUM's
    largest element instead, floored so a collapsing spectrum still gets an
    absolute band (``rcwa/_core._CUT_BAND_REL``, ``elements/berreman.py:188``
    and ``:350``, ``elements/_berreman_jax.py:76``,
    ``elements/eme/eme_2d_vector.py:255``, ``elements/pmm/_core.py:6603``), and
    ``_CUT_BAND_REL``'s own docstring records that the per-mode choice was
    measured against it and REJECTED: "at a cutoff the mode's own magnitude has
    collapsed and judging its real part against it is judging noise against
    noise ... The spectrum's top is the only stable scale there."

    The cylindrical peer of that cutoff is a radial order approaching its own
    cutoff, ``q^2 = k0^2 eps - gamma_j^2 -> 0``, where the discriminating ratio
    ``rho = |Im q| / |Re q| ~ 1 / qn^2`` grows without limit at FIXED backward
    error.  So near a cutoff the classifier reads the eigensolver's backward
    error, the order is called EVANESCENT, and its direction is then taken from
    ``sign(Im q)`` -- which is that same backward error.  When that picks the
    ``Re q < 0`` root the order is shipped BACKWARD, the R/T gate's
    ``qn.real > 1e-6`` leg drops it, and the channel count itself becomes a
    function of the arithmetic.

    MEASURED ON A 39-RUNG NEAR-CUTOFF LADDER (Rbig = 24, N = 120, n = 1.41,
    m = 0/1/2, ``qn`` from 1.4e-02 down to 1.4e-05;
    ``validation/probe_fix_bor_guards/s2_band.py``, re-measuring
    ``validation/probe_scope_bor_guards/a4_*``):

    ==================================  =================  ================
    quantity                            SHIPPED per-mode   THIS spectrum
    ==================================  =================  ================
    worst lossless closure              1.2167e-04         1.9655e-07
    distinct R/T channel counts         {2, 3}             {3}
    ==================================  =================  ================

    619x better on the closure, and the channel count stops moving.  The
    scoping measured the same ladder across three OpenBLAS kernels and two
    builds: the shipped rule's CLASS verdict moved with the kernel on 7 of 24
    rungs and its CHANNEL COUNT on 21 of 24, against 0 of 24 for this scale;
    worst closure over all kernels 2.1579e-04 against 1.2716e-06, 170x.  Under
    the THREAD count alone the shipped rule moved 35 of 39 rungs.  Per
    ``docs/TESTING_STANDARDS.md`` that is a defect, not noise.

    THE FLOOR IS ``k0``, NOT 1.0.  Every Cartesian peer floors its spectrum
    scale at a literal 1.0 because its eigenvalue is dimensionless.  ``q`` here
    carries units of inverse length, so a literal 1.0 would make the band
    unit-system-dependent -- exactly the failure audit P2-06 fixed for the
    channel gate ("absolute thresholds on q silently returned empty R/T for
    small-k0 unit systems").  ``k0`` is the natural non-zero floor of the
    problem and is what was measured.

    THE TWO-SIDED BAR at ``band = 1e-8``, in the discriminating ratio this
    scale defines, ``sigma = |Im q| / max(max|q|, k0)``.  Measured on THIS
    build (Windows py3.14 / numpy 2.4.4 / Haswell, ``OPENBLAS_NUM_THREADS=1``)
    and re-measured on every running build by
    ``test_fix_bor_multilayer_guards.py::test_band_two_sided_population``:

    =========================================  ===  ==============  ==========
    population                                   n  worst ``sigma``  room
    =========================================  ===  ==============  ==========
    NOISE, ordinary lossless geometry             27  1.3024e-15     6.89 dec
    NOISE, deep cutoff (qn to 1.4e-13)            36  8.7301e-10     1.06 dec
    SIGNAL, genuinely lossy Im(n) = 1e-3           6  9.4570e-05     3.98 dec
    SIGNAL, thin end     Im(n) = 1e-6              6  9.4570e-08     0.98 dec
    =========================================  ===  ==============  ==========

    The NOISE rows are the worst ``sigma`` the band must REACH; the SIGNAL rows
    the smallest it must NOT.  The binding side is the deep cutoff at 1.06
    decades: that population is backward error and grows with ``||K||``, so a
    much finer radial grid would eat into it -- which is why the test
    re-measures rather than pins.

    WHY THE THIN SIGNAL END IS ACCEPTABLE.  ``sigma`` is exactly linear in the
    imaginary index, so this band calls media with ``Im(n)`` between ~4e-07 and
    ~1e-09 "propagating" where the old one did not, and orients them by flux
    instead of by decay.  That is a HARMLESSNESS boundary, not a correctness
    one, and it was measured: over **645** physically propagating modes at
    ``Im(n)`` from 1e-06 down to 1e-10 (m = 0, 1, 2) the flux verdict and the
    decay verdict agree on EVERY one -- 0 disagreements -- with the flux at
    ``|P|/fnrm >= 0.1197``, eleven decades above the normalizer's own noise
    fallback.  Where the two rules agree, which one governs cannot matter.

    WHAT IT COST ON ORDINARY GEOMETRY: nothing, measured.  A 30-fixture battery
    (both bases x m = 0,1,2,5 x k0 = 0.8/2.0/3.5 x four geometry families),
    hashed to the SHA-256 of the exact IEEE-754 bytes of R and T: **30 of 30
    bit-identical** across this change on both builds, and all 148 BOR gates
    unchanged.
    """
    if xp is None:
        xp = array_namespace(q)
    # An empty spectrum cannot arise from a real layer, but the fallback stays
    # inside ``xp`` rather than coercing to a Python float: under JAX tracing
    # ``float(k0)`` would raise, and a dead branch that raises is still a trap.
    if getattr(q, "size", 1) == 0:
        return xp.abs(k0)
    return xp.maximum(xp.max(xp.abs(q)), xp.abs(k0))


def forward_orient(q, flux, k0, *, xp=None, band=_BOR_CUT_BAND_REL,
                   scale=None):
    """Forward-orient a whole layer's axial wavenumbers ``q``.

    ``flux`` is the per-mode ``r dr``-weighted axial Poynting flux evaluated at
    the UN-oriented root (the caller has it already; recomputing it here would
    need the caller's grid weights and field blocks, which differ per basis).
    ``k0`` is the vacuum wavenumber and floors the classifier scale.

    Propagating modes (``|Im q| <= band * scale``, :func:`orient_band_scale`)
    are oriented by ``flux >= 0``; the rest by ``Im q > 0``.  Returns the
    oriented ``q``; the caller re-evaluates its fields at the returned root.

    ``xp`` is the array namespace.  The eager paths let it be detected from
    ``q``; the JAX twins pass ``xp=jnp`` EXPLICITLY so the traced body is the
    same object the eager path runs.  Nothing here reads a traced value as a
    Python bool, so the body is ``jit``- and ``grad``-safe.

    ``scale`` may be passed pre-computed when one spectrum is oriented in more
    than one call; it defaults to :func:`orient_band_scale` of ``q``.
    """
    if xp is None:
        xp = array_namespace(q, flux)
    if scale is None:
        scale = orient_band_scale(q, k0, xp=xp)
    # The two shipped shapes of the evanescent leg -- ``not (Im q > 0)`` in the
    # per-mode loops and ``Im q < 0`` in the vectorized copies -- differ only at
    # EXACTLY ``Im q == 0``, which the evanescent leg cannot reach: the band is
    # strictly positive (its floor is 1e-300 times 1e-9, a subnormal but not a
    # zero), so ``|Im q| == 0`` always classifies PROPAGATING.  The vectorized
    # shape is kept.
    prop = xp.abs(xp.imag(q)) <= band * scale
    flip = xp.where(prop, flux < 0.0, xp.imag(q) < 0.0)
    return xp.where(flip, -q, q)


def flux_is_strong(flux, fnrm, *, xp=None, rel=_BOR_FLUX_FALLBACK_REL):
    """``True`` where a mode's ``|z-flux|`` is large enough, RELATIVE to its own
    ``r dr`` field norm, to normalize the mode by flux rather than by norm.

    THE ONE DEFINITION of the flux-normalizer's fallback decision, which had
    six copies (``zcascade.py:96``, ``sem_radial.py:436``, ``bor_solve.py:51``,
    ``_jax_bor.py:108``, ``_jax_sem.py:255``, plus the comment at
    ``bor_stack.py:678`` that documents it).  What is shared -- and what could
    silently drift -- is the PREDICATE and its constant; the SCALE each site
    then applies is deliberately left at the call site, because the five sites
    normalize against genuinely different fallback measures (a two-grid
    ``r dr`` norm on the SEM basis, a plain column 2-norm on the FD basis) and
    folding them together would change numbers rather than deduplicate a
    decision.  That is the same resolution audit S1-16 reached for the channel
    gate: share the core, keep each basis's own leg where it belongs, and say
    so.
    """
    if xp is None:
        xp = array_namespace(flux, fnrm)
    return xp.abs(flux) > rel * xp.real(fnrm)


def channel_core(qn, *, xp=None, imag_bar=_BOR_CHANNEL_IMAG_BAR,
                 real_floor=_BOR_CHANNEL_REAL_FLOOR):
    """THE ONE ``{imag, real-floor}`` core of the R/T channel gate, on the
    DIMENSIONLESS axial index ``qn = q / k0``.

    Five copies (``bor_solve.py:180``, ``bor_stack.py:692``,
    ``bor_stack.py:890``, ``_jax_bor.py:195``, ``_jax_sem.py:390``) carried
    this pair of comparisons.  Each call site keeps its OWN basis-specific leg
    and ``&``-s it with this -- the legacy nodal basis its divergence tag
    (``reldiv < tol``, which screens the spurious sea that basis grows), the
    staggered / SEM twins the index ceiling (``Re sqrt(eps) - Re qn >
    -5e-10``).  Audit S1-16 documents and justifies that split: forcing the
    ceiling onto the nodal basis over-filters its reldiv-screened set and
    degrades its documented ~4% energy floor (measured 4% -> 10.7%), and the
    staggered bases are div-conforming so ``reldiv`` is structurally zero and
    its eigensolve is deliberately skipped.  The split is real physics; the
    two comparisons above it were pure duplication.
    """
    if xp is None:
        xp = array_namespace(qn)
    return (xp.abs(xp.imag(qn)) < imag_bar) & (xp.real(qn) > real_floor)
