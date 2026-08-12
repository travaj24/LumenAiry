# validation/pipeline/spec.py -- the CONFIGURATION of a staged field-
# aggregation run, as data.
#
# WHY A SPEC OBJECT AND NOT KEYWORD ARGUMENTS.  A staged pipeline is
# resumable, which means a stage that runs today has to be able to prove that
# the artifact it is about to reuse was produced by the SAME configuration.
# That is only possible if the configuration is a VALUE: serialisable,
# hashable, and complete.  ``_d121_common._chain_a_key`` is the same argument
# made one artifact at a time; this module makes it for the whole run.
#
# NO MAGIC, and that is enforced rather than promised:
#
#   * every field is typed and defaulted HERE, so a spec file that omits a
#     field gets a default that is written down in one place;
#   * an UNKNOWN key is refused, not ignored -- a typo in a config file is
#     otherwise a silent revert to the default, which is the exact class of
#     "plausible-looking wrong answer" this campaign refuses;
#   * every enum is checked against its own vocabulary at construction, not at
#     the point of use three stages later;
#   * ``to_dict``/``from_dict`` round-trip EXACTLY (floats via repr), so the
#     spec stored beside an artifact is the spec, not a rendering of it.
#
# The plug-in points (which decomposer, which chain runner) are named by
# STRING and looked up in a registry, so a spec file is a complete description
# of a run and never a Python import path that has to be trusted.
#
# cp1252-safe ASCII only.
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

#: Bumped when the spec vocabulary changes in a way that would make a stored
#: spec mean something different.  Artifacts carry it; a mismatch orphans them
#: rather than being silently re-interpreted.
PIPELINE_SCHEMA = 1

#: The stage names, IN ORDER.  This tuple is the pipeline: everything else
#: (resume, --from, the manifest) is indexed by it.
STAGES = ('decompose', 'chains', 'aggregate', 'leg', 'readout')

_GUARD_ACTIONS = ('error', 'warn', 'ignore')
_BOX_ACTIONS = ('warn', 'error')
_LEG_MODES = ('crop', 'full')
_PLANES = ('fine_retrace_exit',)
_WEIGHT_SOURCES = ('decompose', 'unit')


class SpecError(ValueError):
    """A configuration that cannot describe a run.

    Distinct from ``ValueError`` at the call sites so a driver can tell a bad
    CONFIG (fix the file) from a bad RUN (fix the box or the physics)."""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _enc(v):
    """JSON-safe encoding that round-trips a float bit-exactly.

    Finite floats go out as floats (``repr`` round-trips in float64); a
    non-finite one goes out as its ``repr`` STRING, because ``inf`` is the
    library's own spelling of a collimated carrier and bare ``Infinity`` is
    not JSON."""
    if isinstance(v, float):
        return v if math.isfinite(v) else repr(v)
    if isinstance(v, tuple):
        return [_enc(x) for x in v]
    if isinstance(v, list):
        return [_enc(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _enc(x) for k, x in v.items()}
    return v


def _num(v):
    return float(v) if not isinstance(v, str) else float(v)


def _check_enum(name, value, allowed, where):
    if value not in allowed:
        raise SpecError(
            f"{where}.{name}: {value!r} is not one of "
            f"{', '.join(repr(a) for a in allowed)}.")


def _check_positive(name, value, where, strict=True):
    v = float(value)
    if not math.isfinite(v) or (v <= 0.0 if strict else v < 0.0):
        raise SpecError(
            f"{where}.{name} must be a finite "
            f"{'positive' if strict else 'non-negative'} number, got "
            f"{value!r}.")
    return v


def _from_mapping(cls, d, where):
    """Build a dataclass from a mapping, REFUSING unknown keys.

    The refusal is the point.  A config file that says ``nyquist_margins``
    would otherwise silently run at the default 1.0 while its author believed
    it ran at 2.0, and nothing downstream could tell."""
    if not isinstance(d, dict):
        raise SpecError(f"{where}: expected a mapping, got "
                        f"{type(d).__name__}.")
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(d) - known)
    if unknown:
        raise SpecError(
            f"{where}: unknown key(s) {', '.join(repr(k) for k in unknown)}.  "
            f"Known keys are {', '.join(sorted(known))}.  A config key that "
            f"is silently ignored is a run that did not do what its file "
            f"says, so it is refused.")
    return cls(**d)


# ---------------------------------------------------------------------------
# stage specs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DecomposeSpec:
    """Which decomposer builds the beam list, and with what parameters.

    ``kind`` is a registry name (see ``sources.register_decomposer``), never
    an import path: a spec file describes a run, it does not get to name code
    to import."""

    kind: str
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.kind, str) or not self.kind:
            raise SpecError("decompose.kind must be a non-empty registry "
                            "name.")
        if not isinstance(self.params, dict):
            raise SpecError(f"decompose.params must be a mapping, got "
                            f"{type(self.params).__name__}.")


@dataclass(frozen=True)
class ChainSpec:
    """One beam's chain to the NAMED plane, and the grid of record.

    ``plane`` is spelled out and validated even though there is exactly one
    legal value today.  PROBE_SUM_AT_APERTURE S2 is the reason: the only plane
    at which traced congruences may be added is the last group's back aperture
    ON THE EXACT LEG'S FINE RETRACE GRID.  The coarse co-moving exit vertex is
    7.8x under-sampled on its own exit sphere and anything before the last
    group cannot be traced at all.  Naming the plane in the config makes that
    a decision a reader can see, and the enum makes a second value an explicit
    act rather than a typo.

    ``ray_subsample`` DEFAULTS TO 1, not to the library's 4.
    PROBE_CHAIN_LADDER_PISTON S2.4 measured that ``rs = 4`` loses 6.5e-03 of
    the aperture power -- 160x this campaign's 4e-05 energy bar -- and that
    ``rs = 1`` at N = 1024 recovers it (99.979 % vs 99.353 %) in 113 s where
    ``N = 4096`` at ``rs = 4`` reaches only 99.953 % in 400 s.  An aggregating
    consumer is exactly the consumer for which that power matters, so the
    default here is the one that measurement recommends.  A run that must
    reproduce an ARCHIVED artifact sets it back to whatever produced that
    artifact and says so."""

    kind: str = 'traced'
    plane: str = 'fine_retrace_exit'
    ray_subsample: int = 1
    n_workers: int = 1
    n_fine_cap: int = 8192
    window_factor: float = 4.0
    final_leg: str = 'auto'
    ram_budget: str = 'inf'
    on_box_budget: str = 'warn'
    capture_reference_tile: bool = True
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.kind, str) or not self.kind:
            raise SpecError("chain.kind must be a non-empty registry name.")
        _check_enum('plane', self.plane, _PLANES, 'chain')
        _check_enum('on_box_budget', self.on_box_budget, _BOX_ACTIONS, 'chain')
        for nm in ('ray_subsample', 'n_workers', 'n_fine_cap'):
            v = getattr(self, nm)
            if not isinstance(v, int) or v < 1:
                raise SpecError(f"chain.{nm} must be an integer >= 1, got "
                                f"{v!r}.")
        _check_positive('window_factor', self.window_factor, 'chain')
        if not isinstance(self.params, dict):
            raise SpecError("chain.params must be a mapping.")


@dataclass(frozen=True)
class AggregateSpec:
    """The common lattice, the common carrier, and which beams enter the sum.

    ``include`` is separate from the decomposer's beam list ON PURPOSE.  The
    probe's crosstalk census (PROBE_SUM_AT_APERTURE S6.4) is exactly this:
    sum ONE order and read ALL the frames, so the off-diagonal is the power
    frame i receives from order j.  The shipped per-order path cannot express
    that at all -- its off-diagonal is zero by construction, not by physics.

    ``batch_size`` bounds peak memory: a design-121 aperture envelope is
    1.07 GB, so 32 of them resident is 34 GB before any working set.  Fields
    are loaded, re-referenced, accumulated and released in batches of this
    size.  ``batch_size >= len(include)`` is ONE ``aggregate`` call and is
    therefore bit-identical to the un-batched primitive; a smaller batch
    changes only the ORDER of a float64 summation (the probe measured
    aggregation linearity at 2.3e-16)."""

    dx_common: float
    n_common: int
    origin: Tuple[float, float] = (0.0, 0.0)
    reference_beam: Optional[str] = None
    include: Union[str, List[str]] = 'all'
    weights: str = 'decompose'
    batch_size: int = 8
    support_frac: float = 0.99999
    nyquist_margin: float = 1.0
    on_nyquist: str = 'error'
    on_window: str = 'warn'
    bandlimit: bool = False

    def __post_init__(self):
        _check_positive('dx_common', self.dx_common, 'aggregate')
        if not isinstance(self.n_common, int) or self.n_common < 2:
            raise SpecError(f"aggregate.n_common must be an integer >= 2, got "
                            f"{self.n_common!r}.")
        org = tuple(float(v) for v in self.origin)
        if len(org) != 2 or not all(math.isfinite(v) for v in org):
            raise SpecError(f"aggregate.origin must be two finite metres, got "
                            f"{self.origin!r}.")
        object.__setattr__(self, 'origin', org)
        _check_enum('weights', self.weights, _WEIGHT_SOURCES, 'aggregate')
        _check_enum('on_nyquist', self.on_nyquist, _GUARD_ACTIONS, 'aggregate')
        _check_enum('on_window', self.on_window, _GUARD_ACTIONS, 'aggregate')
        if not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise SpecError(f"aggregate.batch_size must be an integer >= 1, "
                            f"got {self.batch_size!r}.")
        f = _check_positive('support_frac', self.support_frac, 'aggregate')
        if not (0.0 < f < 1.0):
            raise SpecError(f"aggregate.support_frac must lie strictly in "
                            f"(0, 1) -- it is an ENCLOSED-POWER fraction, not "
                            f"a percentage -- got {self.support_frac!r}.")
        m = _check_positive('nyquist_margin', self.nyquist_margin, 'aggregate')
        if m < 1.0:
            raise SpecError(
                f"aggregate.nyquist_margin = {m!r} is BELOW bare Nyquist.  "
                f"1.0 is the sampling theorem; a value under it asks the "
                f"guard to accept a grid that provably cannot carry the "
                f"carrier difference.  Use on_nyquist='warn' if the aliased "
                f"skirt is genuinely below your bar, so the decision is "
                f"recorded as a disposition rather than hidden in a margin.")
        if not (self.include == 'all' or (
                isinstance(self.include, (list, tuple))
                and all(isinstance(k, str) for k in self.include))):
            raise SpecError(f"aggregate.include must be 'all' or a list of "
                            f"beam keys, got {self.include!r}.")
        if isinstance(self.include, tuple):
            object.__setattr__(self, 'include', list(self.include))
        if isinstance(self.include, list) and not self.include:
            raise SpecError("aggregate.include is an empty list -- an "
                            "aggregation of nothing has no field.")


@dataclass(frozen=True)
class LegSpec:
    """The ONE exact final leg run on the summed field.

    ``mode``:

    * ``'crop'`` -- crop the sum about THIS frame's chief ray to the same
      physical window and the same internal fine grid the shipped per-order
      path used, still referencing the ONE common sphere.  Legal precisely
      because every order shares ``R`` at this plane.  This is the
      LIKE-FOR-LIKE variant and it is what the probe's scored tables use.
    * ``'full'`` -- one leg on the whole summed aperture, as the architecture
      specifies.  Cleaner physics for crosstalk (it truncates nothing), a
      different window from the shipped path, and more expensive.

    NOTE ON THE NAME 'single leg'.  No library entry point separates the
    shareable part of an exact leg (sphere reference, crop/upsample,
    reconstruct, forward FFT, transfer function -- ~81 %) from the
    irreducibly per-frame Bluestein zoom (~19 %); see
    PROBE_SUM_AT_APERTURE S8.3 and blocker 3 of its S9.  So this stage
    RESOLVES one leg plan against one summed field and one common carrier,
    and the readout stage then executes it once per frame.  The physics is
    the single-leg architecture; the cost is not yet, and this pipeline does
    not pretend otherwise."""

    distance: float
    mode: str = 'crop'
    n_fine_cap: int = 8192
    window_factor: float = 4.0
    ram_budget: str = 'inf'
    on_replica: str = 'warn'
    on_readout_window: str = 'warn'
    on_ram_cap: str = 'error'
    on_n_fine_cap: str = 'error'

    def __post_init__(self):
        _check_enum('mode', self.mode, _LEG_MODES, 'leg')
        for nm in ('on_replica', 'on_readout_window', 'on_ram_cap',
                   'on_n_fine_cap'):
            _check_enum(nm, getattr(self, nm), _GUARD_ACTIONS, 'leg')
        d = float(self.distance)
        if not math.isfinite(d):
            raise SpecError(f"leg.distance must be a finite propagation "
                            f"distance in metres, got {self.distance!r}.")
        object.__setattr__(self, 'distance', d)
        if not isinstance(self.n_fine_cap, int) or self.n_fine_cap < 2:
            raise SpecError(f"leg.n_fine_cap must be an integer >= 2, got "
                            f"{self.n_fine_cap!r}.")
        _check_positive('window_factor', self.window_factor, 'leg')


@dataclass(frozen=True)
class ReadoutSpec:
    """The per-frame Bluestein zoom plan.

    ``frames`` defaults to ``'all'`` -- every beam the decomposer produced,
    whether or not it entered the sum.  That is what makes a crosstalk census
    a single run rather than a special mode."""

    dx_out: float
    n_out: int
    frames: Union[str, List[str]] = 'all'
    save_frames: bool = True
    metrics: bool = True
    ee_radii: List[float] = field(default_factory=lambda: [3e-6, 6e-6, 12e-6])

    def __post_init__(self):
        _check_positive('dx_out', self.dx_out, 'readout')
        if not isinstance(self.n_out, int) or self.n_out < 2:
            raise SpecError(f"readout.n_out must be an integer >= 2, got "
                            f"{self.n_out!r}.")
        if not (self.frames == 'all' or (
                isinstance(self.frames, (list, tuple))
                and all(isinstance(k, str) for k in self.frames))):
            raise SpecError(f"readout.frames must be 'all' or a list of beam "
                            f"keys, got {self.frames!r}.")
        if isinstance(self.frames, tuple):
            object.__setattr__(self, 'frames', list(self.frames))
        rr = [float(r) for r in self.ee_radii]
        if not rr or any(not math.isfinite(r) or r <= 0.0 for r in rr):
            raise SpecError(f"readout.ee_radii must be positive metres, got "
                            f"{self.ee_radii!r}.")
        object.__setattr__(self, 'ee_radii', rr)


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PipelineSpec:
    """A complete, serialisable description of one staged run.

    ``coherent_sum`` is a DECLARATION, not a switch that changes arithmetic.
    Summing fields across chains is only admissible once the chains return an
    ABSOLUTE optical path.  On a library WITHOUT
    ``FIX_TILT_QUADRATIC_OPL_2026_08_11`` the inter-chain piston is
    reproducible and WRONG by 0.050-0.416 waves (PROBE_CHAIN_LADDER_PISTON
    S3.6/S3.8) -- 5x to 42x outside lambda/100 -- so a coherent add imposes an
    essentially arbitrary phase per beam.  Setting ``coherent_sum=True`` on
    such a library gets a loud runtime warning and the piston columns of every
    report are flagged; the intensity columns are unaffected, because the
    orders land on separate frames and the defect is a per-beam constant."""

    name: str
    workdir: str
    wavelength: float
    decompose: DecomposeSpec
    chain: ChainSpec
    aggregate: AggregateSpec
    leg: LegSpec
    readout: ReadoutSpec
    coherent_sum: bool = False
    notes: str = ''

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise SpecError("name must be a non-empty string -- it labels "
                            "every artifact this run writes.")
        if not isinstance(self.workdir, str) or not self.workdir.strip():
            raise SpecError("workdir must be a non-empty path.")
        lam = _check_positive('wavelength', self.wavelength, 'spec')
        object.__setattr__(self, 'wavelength', lam)
        for nm, typ in (('decompose', DecomposeSpec), ('chain', ChainSpec),
                        ('aggregate', AggregateSpec), ('leg', LegSpec),
                        ('readout', ReadoutSpec)):
            v = getattr(self, nm)
            if not isinstance(v, typ):
                raise SpecError(f"{nm} must be a {typ.__name__}, got "
                                f"{type(v).__name__}.  Build it from a "
                                f"mapping with PipelineSpec.from_dict.")
        if not isinstance(self.coherent_sum, bool):
            raise SpecError(f"coherent_sum must be a bool, got "
                            f"{self.coherent_sum!r}.")

    # -- serialisation ----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        d = {'schema': PIPELINE_SCHEMA}
        for f in fields(self):
            v = getattr(self, f.name)
            d[f.name] = _enc(asdict(v)) if is_dataclass(v) else _enc(v)
        return d

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=1,
                          allow_nan=False, **kw)

    @classmethod
    def from_dict(cls, d) -> 'PipelineSpec':
        if not isinstance(d, dict):
            raise SpecError(f"PipelineSpec.from_dict: expected a mapping, got "
                            f"{type(d).__name__}.")
        d = dict(d)
        schema = int(d.pop('schema', PIPELINE_SCHEMA))
        if schema != PIPELINE_SCHEMA:
            raise SpecError(
                f"PipelineSpec: this spec carries schema {schema} but this "
                f"build reads schema {PIPELINE_SCHEMA}.  Refusing to guess: a "
                f"re-interpreted field is a run that silently did something "
                f"else.")
        sub = {'decompose': DecomposeSpec, 'chain': ChainSpec,
               'aggregate': AggregateSpec, 'leg': LegSpec,
               'readout': ReadoutSpec}
        kw: Dict[str, Any] = {}
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(d) - known)
        if unknown:
            raise SpecError(
                f"PipelineSpec: unknown top-level key(s) "
                f"{', '.join(repr(k) for k in unknown)}.  Known keys are "
                f"{', '.join(sorted(known))}.")
        for k, v in d.items():
            if k in sub:
                kw[k] = _from_mapping(sub[k], v, k)
            else:
                kw[k] = v
        missing = sorted(k for k in sub if k not in kw)
        if missing:
            raise SpecError(f"PipelineSpec: missing required section(s) "
                            f"{', '.join(missing)}.")
        return cls(**kw)

    @classmethod
    def from_json(cls, text) -> 'PipelineSpec':
        return cls.from_dict(json.loads(text))

    @classmethod
    def load(cls, path) -> 'PipelineSpec':
        with open(path, 'r', encoding='cp1252', errors='strict') as fh:
            return cls.from_json(fh.read())

    def replace(self, **kw) -> 'PipelineSpec':
        """A copy with top-level fields replaced.  Sub-specs are values, so
        ``spec.replace(aggregate=spec.aggregate.__class__(...))`` is how a
        driver derives an arm without mutating the spec of record."""
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d.update(kw)
        return PipelineSpec(**d)


def resolve_selection(selection: Union[str, Sequence[str]],
                      available: Sequence[str], where: str) -> List[str]:
    """Turn ``'all'`` or a list of keys into a concrete, ORDER-PRESERVING list.

    Order is preserved from ``available`` rather than from the selection, so
    the accumulation order of a sum is a property of the beam list and not of
    how a config file happened to spell its subset.  A key that is not
    available is an error, never a silent drop: a 3-order run that quietly
    became a 2-order run is a wrong answer with no signature."""
    if selection == 'all':
        return list(available)
    want = list(selection)
    missing = [k for k in want if k not in available]
    if missing:
        raise SpecError(
            f"{where}: beam key(s) {', '.join(repr(k) for k in missing)} are "
            f"not in the decomposed beam list "
            f"({', '.join(repr(k) for k in available)}).")
    seen = set(want)
    return [k for k in available if k in seen]
