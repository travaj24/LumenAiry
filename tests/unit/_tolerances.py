"""Named test-tolerance classes — see docs/TOLERANCE_POLICY.md for the
decision rule (when bit-identity is allowed vs when a physical tolerance is
mandatory).  New tests should import these instead of inventing numbers;
existing tests migrate opportunistically when touched."""

# cross-path agreement through degenerate eigendecompositions
# (stack <-> segments, generalized <-> symmetric cascade, promotion <-> direct)
GAUGE_CROSS_PATH = 5e-6

# contracts evaluated AT a degenerate limit point (slant -> 0, off-plane -> 0)
GAUGE_AT_LIMIT = 1e-4

# R+T-1 on a provably-lossless single solve, clean geometry
LOSSLESS_CLOSURE = 1e-9

# R+T-1 through many-layer staircases (accumulated Redheffer roundoff)
STAIRCASE_CLOSURE = 1e-7

# a stabilized answer vs the adjacent-truncation consensus (per order)
CONSENSUS_PER_ORDER = 2e-4

# PMM <-> RCWA cross-family agreement
CROSS_FAMILY_DIELECTRIC = 2e-3
CROSS_FAMILY_METAL = 2e-2
