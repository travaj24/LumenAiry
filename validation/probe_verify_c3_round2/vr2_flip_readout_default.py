"""VERIFY-WP-C3 ROUND 2, item 2 -- a pytest plugin that FLIPS
``carrier_referenced_focus_readout``'s new ``transport`` default from
``'sziklas'`` to ``'collins'`` in place, so the round-2 claim "flipping the
default turns 8 a6 ids red" can be measured without editing ``lumenairy/``.

Usage:  pytest -p vr2_flip_readout_default <files>
(with this directory and the tree root on PYTHONPATH).
"""
import lumenairy.propagators.carrier as C

_FN = C.carrier_referenced_focus_readout
assert _FN.__kwdefaults__['transport'] == 'sziklas'
_FN.__kwdefaults__['transport'] = 'collins'
print('[vr2] carrier_referenced_focus_readout transport default ->',
      _FN.__kwdefaults__['transport'])
