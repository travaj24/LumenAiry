# validation/pipeline/run_pipeline.py -- the CLI.
#
#   python run_pipeline.py <spec.json>
#       [--from decompose|chains|aggregate|leg|readout]   recompute from here
#       [--through <stage>]                               stop after here
#       [--only <stage>[,<stage>...]]                     run exactly these
#       [--set a.b=<json> ...]                            override a field
#       [--print-spec]                                    resolve and exit
#
# ``--set`` takes a DOTTED PATH and a JSON value, so an arm of a study is a
# command line and not a second copy of a spec file:
#
#   --set aggregate.include='["p0_p0"]'   --set workdir='"..."'
#
# The resolved spec is written to ``<workdir>/spec.json`` on every run, so the
# artifact directory always carries the configuration that produced it --
# including the overrides.
#
# cp1252-safe ASCII only.
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
# ``python <script>.py`` puts the SCRIPT's directory on sys.path, not the cwd,
# so an unrelated pip-installed lumenairy would win over the checkout this
# script belongs to.  Pin the checkout, in front.
for _p in (_ROOT, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# The two known-noisy, understood categories the design-121 runners silence --
# and NOTHING else.  ``fan_multi_121.py``'s header records why a blanket
# filter is refused here: the first cut of that runner silenced everything and
# hid the only signal that the readout window was in the periodic-replica
# regime, which is the exact class of silent wrong answer these runs exist to
# detect.
warnings.filterwarnings('ignore', message='.*prescription aperture.*')
warnings.filterwarnings('ignore', message='.*residual transverse.*')
warnings.filterwarnings('ignore', message='.*under-sampled.*')

from pipeline import PipelineSpec, run              # noqa: E402
from pipeline.spec import SpecError                 # noqa: E402


def apply_override(doc, path, value):
    """``a.b.c=<json>`` into a nested dict, refusing to CREATE a section.

    Creating one would let a typo introduce a key the spec's own validation
    then rejects with a confusing message three levels down -- or worse,
    silently populate ``params``, which is the one place unknown keys are
    legal."""
    parts = path.split('.')
    d = doc
    for p in parts[:-1]:
        if p not in d or not isinstance(d[p], dict):
            raise SpecError(
                f"--set {path}: {p!r} is not an existing section of this spec "
                f"(have {sorted(k for k, v in d.items() if isinstance(v, dict))}"
                f").")
        d = d[p]
    d[parts[-1]] = value
    return doc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('spec')
    ap.add_argument('--from', dest='from_stage', default=None)
    ap.add_argument('--through', default=None)
    ap.add_argument('--only', default=None)
    ap.add_argument('--set', dest='sets', action='append', default=[])
    ap.add_argument('--print-spec', action='store_true')
    a = ap.parse_args(argv)

    with open(a.spec, 'r', encoding='cp1252', errors='strict') as fh:
        doc = json.load(fh)
    for s in a.sets:
        if '=' not in s:
            raise SystemExit(f"--set expects key=<json>, got {s!r}")
        k, v = s.split('=', 1)
        try:
            val = json.loads(v)
        except ValueError:
            raise SystemExit(
                f"--set {k}: {v!r} is not JSON.  Strings need quotes "
                f"(e.g. --set leg.mode='\"full\"').")
        apply_override(doc, k.strip(), val)
    # Relative workdirs resolve against the SPEC FILE, not the cwd: a spec is
    # a portable description of a run, and a run that lands somewhere else
    # depending on where it was launched from is a run whose artifacts cannot
    # be found by the next person.
    if not os.path.isabs(doc.get('workdir', '')):
        doc['workdir'] = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(a.spec)),
                         doc['workdir']))
    spec = PipelineSpec.from_dict(doc)
    if a.print_spec:
        print(spec.to_json())
        return 0
    only = [s.strip() for s in a.only.split(',')] if a.only else None
    run(spec, from_stage=a.from_stage, through=a.through, only=only)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
