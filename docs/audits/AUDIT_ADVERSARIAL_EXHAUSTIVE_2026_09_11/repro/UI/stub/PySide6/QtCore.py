"""Minimal PySide6.QtCore stub sufficient to import lumenairy.ui.model headlessly."""


class _BoundSignal:
    def __init__(self):
        self._slots = []

    def connect(self, slot):
        self._slots.append(slot)

    def disconnect(self, slot=None):
        if slot is None:
            self._slots.clear()
        elif slot in self._slots:
            self._slots.remove(slot)

    def emit(self, *args):
        for s in list(self._slots):
            s(*args)


class Signal:
    def __init__(self, *types, **kw):
        self._types = types
        self._name = None

    def __set_name__(self, owner, name):
        self._name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        store = obj.__dict__.setdefault('_stub_signals', {})
        key = self._name or id(self)
        if key not in store:
            store[key] = _BoundSignal()
        return store[key]


class QObject:
    def __init__(self, parent=None):
        self._parent = parent

    def parent(self):
        return self._parent


class QTimer:
    def __init__(self, *a, **k):
        pass

    @staticmethod
    def singleShot(ms, fn):
        fn()


class Qt:
    pass


class QThread(QObject):
    pass
