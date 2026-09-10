"""ITEM 4: narrow the two ``except Exception:`` blocks added this wave under
``lumenairy/elements/pmm/_core.py``.

Both are diagnostic-only guards, so the fix is to name what they actually
catch, NOT to widen the budget (48).

  site 1  ``_stag_grid_text``  -- a formatter for an error message.  It reads
          attributes off a StagGridOps (``AttributeError`` if the object is a
          different shape), calls ``np.asarray`` / ``np.diff`` / ``np.min`` on
          ``b.xb`` (``TypeError`` on a non-array-like, ``ValueError`` from
          ``np.min`` of an EMPTY segment array), and divides by ``b.d``
          (``ZeroDivisionError`` on a zero period).

  site 2  the ``gecon`` condition estimate.  ``get_lapack_funcs`` raises
          ``ValueError`` for an unsupported dtype and ``AttributeError`` when
          the routine is absent from the linked LAPACK; the wrapper itself
          raises ``ValueError`` / ``TypeError`` on a shape or dtype mismatch
          and ``LinAlgError`` on a degenerate factor set; ``float(rcv)`` raises
          ``TypeError`` if the wrapper hands back something unexpected.  The
          estimate is an INSTRUMENT -- a failure must leave ``rc`` NaN and let
          the caller's own screen decide, never mask a real error.
"""
import io
import sys

p = "lumenairy/elements/pmm/_core.py"
s = io.open(p, encoding="cp1252").read()

old1 = '''        return f"    grid {label}: M={g.M}, q={g.q}; " + "; ".join(bits)
    except Exception:                       # noqa: BLE001  (diagnostic only)
        return f"    grid {label}: <undescribable>"
'''
new1 = '''        return f"    grid {label}: M={g.M}, q={g.q}; " + "; ".join(bits)
    except (AttributeError, TypeError, ValueError, ZeroDivisionError):
        # AttributeError: not a StagGridOps-shaped object.  TypeError: ``b.xb``
        # is not array-like.  ValueError: ``np.min`` of an EMPTY segment array.
        # ZeroDivisionError: a zero period ``b.d``.  This function only builds
        # a line of an error message, so any of those must degrade the message
        # rather than replace the caller's exception with this one.
        return f"    grid {label}: <undescribable>"
'''

old2 = '''            gecon = sla.get_lapack_funcs("gecon", (A,))
            rcv, info = gecon(lu, anorm)
            rc = float(rcv) if int(info) == 0 else 0.0
        except Exception:                   # noqa: BLE001  (instrument only)
            rc = float("nan")
'''
new2 = '''            gecon = sla.get_lapack_funcs("gecon", (A,))
            rcv, info = gecon(lu, anorm)
            rc = float(rcv) if int(info) == 0 else 0.0
        except (AttributeError, TypeError, ValueError, sla.LinAlgError,
                np.linalg.LinAlgError):
            # AttributeError: ``gecon`` absent from the linked LAPACK.
            # ValueError: an unsupported dtype reaches ``get_lapack_funcs``, or
            # a shape/dtype mismatch reaches the wrapper.  TypeError: the
            # wrapper returns something ``float()`` cannot take.  LinAlgError:
            # a degenerate factor set.  The estimate is an INSTRUMENT: NaN
            # here means "unmeasured", and the screens below are written so a
            # NaN is NOT treated as a refusal.
            rc = float("nan")
'''

if "--revert" in sys.argv:
    old1, new1, old2, new2 = new1, old1, new2, old2

assert s.count(old1) == 1, ("site1", s.count(old1))
assert s.count(old2) == 1, ("site2", s.count(old2))
s = s.replace(old1, new1).replace(old2, new2)
io.open(p, "w", encoding="cp1252", newline="").write(s)
print("reverted" if "--revert" in sys.argv else "patched", p)
