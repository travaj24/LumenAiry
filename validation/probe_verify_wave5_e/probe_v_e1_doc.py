"""VERIFY-WAVE5-E / E1(e): is D4's sentence really in all three places?

Checked against the RUNNING library (the registered knob's doc through the
public knob registry, the setter's ``__doc__``) and the module SOURCE (the
``_PYFFTW_DOUBLE_BUFFER`` note), after collapsing whitespace and stripping
comment prefixes / emphasis markers -- so "verbatim" means the words, not the
line breaks.
"""
import inspect
import json
import re
import sys

import lumenairy
from lumenairy.propagators import fft_infra as fi

SENT = ("the transform's values are byte-identical either way; the object "
        "handed back is a live workspace view in one mode and a private copy "
        "in the other, which NumPy's temporary elision can distinguish")


def norm(t):
    t = t.replace('**', '').replace('``', '')
    t = re.sub(r'^\s*#:?\s?', '', t, flags=re.M)
    return re.sub(r'\s+', ' ', t).strip().lower()


target = norm(SENT)

# 1. the registered knob doc, through the public knob registry
from lumenairy import _knobs as _kn

fi.get_fft_double_buffer()            # force the module's registrations
assert 'fft_double_buffer' in _kn.knobs(), sorted(_kn.knobs())
knob_doc = _kn.knob_doc('fft_double_buffer')

src = inspect.getsource(fi)
i = src.index('_PYFFTW_DOUBLE_BUFFER = True')
module_note = src[max(0, i - 4000):i]

places = {
    'registered_knob_doc': knob_doc,
    'set_fft_double_buffer.__doc__': fi.set_fft_double_buffer.__doc__ or '',
    'module_note__PYFFTW_DOUBLE_BUFFER': module_note,
    'set_fft_plan_max_bytes_per_buffer.__doc__':
        fi.set_fft_plan_max_bytes_per_buffer.__doc__ or '',
}
res = {k: (target in norm(v)) for k, v in places.items()}
print(json.dumps(dict(
    lumenairy_file=lumenairy.__file__, python=sys.version.split()[0],
    sentence=SENT, verbatim_in=res,
    n_verbatim=sum(res.values()),
    knob_doc_found=knob_doc is not None,
    knob_doc_excerpt=norm(knob_doc)[:600] if knob_doc else None,
), indent=1))
