"""
Internal math primitives shared across lumenairy submodules.

This package hosts pure-math helpers that have no domain-specific
content (no notion of wavefronts, prescriptions, rays, etc.) and that
were previously inlined inside the consumer modules that needed them.
Moving them here breaks inverted dependencies where ``propagators/``
modules had to reach into ``elements/`` to get a math primitive.

Subpackages
-----------
chebyshev
    Vandermonde-like tables of Chebyshev polynomials of the first
    kind and their first / second derivatives.  Backend-aware via the
    optional ``xp`` kwarg.

Notes
-----
This is a *private* subpackage (leading underscore on the directory
name).  Public APIs continue to live in their topical modules.  The
extraction here is purely an internal-layering cleanup; existing
``from lumenairy.elements.lenses import _chebyshev_vandermonde``
imports continue to work via back-compat aliases.
"""
