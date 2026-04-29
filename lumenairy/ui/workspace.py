"""
Workspace system — top-of-window tabs that group docks by topic.

Each workspace is a named (title, dock_names, saved_state) triple.  When
the user switches tabs we hide any docks that do not belong to the new
workspace and (optionally) restore a previously-saved QMainWindow layout
so the active docks snap back to where the user last dragged them.

A user can create, rename, duplicate, and delete workspaces, and pick the
docks belonging to each via a checklist dialog.  Defaults cover Design,
Analysis, Wave Optics, Tolerancing, and Materials.

Author: Andrew Traverso
"""

from PySide6.QtCore import Qt, QByteArray, QObject, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QMenu, QPushButton, QTabBar, QToolBar, QVBoxLayout,
)

import base64
import json


# ---------------------------------------------------------------------------
#  Workspace container
# ---------------------------------------------------------------------------

class Workspace:
    """A named dock layout: which docks are visible + a saved state blob."""

    def __init__(self, name, dock_names=None, state_b64=None):
        self.name = str(name)
        self.dock_names = list(dock_names or [])
        self.state_b64 = state_b64  # base64(QMainWindow.saveState())

    def to_dict(self):
        return {
            'name': self.name,
            'dock_names': list(self.dock_names),
            'state_b64': self.state_b64 or '',
        }

    @staticmethod
    def from_dict(d):
        return Workspace(
            name=d.get('name', 'Untitled'),
            dock_names=list(d.get('dock_names') or []),
            state_b64=d.get('state_b64') or None,
        )


# ---------------------------------------------------------------------------
#  Top-of-window tab bar
# ---------------------------------------------------------------------------

class WorkspaceBar(QToolBar):
    """Horizontal tab strip at the top of the window, plus a `+` button."""

    switched = Signal(int)
    add_requested = Signal()
    rename_requested = Signal(int)
    delete_requested = Signal(int)
    duplicate_requested = Signal(int)
    manage_requested = Signal(int)

    def __init__(self, parent=None):
        super().__init__('Workspaces', parent)
        self.setObjectName('workspace_bar')
        self.setMovable(False)
        self.setFloatable(False)

        self.tab_bar = QTabBar()
        self.tab_bar.setExpanding(False)
        self.tab_bar.setDrawBase(False)
        self.tab_bar.setUsesScrollButtons(True)
        self.tab_bar.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tab_bar.currentChanged.connect(self.switched.emit)
        self.tab_bar.customContextMenuRequested.connect(self._on_context_menu)
        self.tab_bar.tabBarDoubleClicked.connect(self.rename_requested.emit)
        self.addWidget(self.tab_bar)

        self._plus_action = QAction('＋', self)   # fullwidth plus
        self._plus_action.setToolTip('New workspace')
        self._plus_action.triggered.connect(self.add_requested.emit)
        self.addAction(self._plus_action)

    def set_names(self, names, current_index=0):
        """Replace the tabs with `names`, then set `current_index` selected.
        Emits `switched` only if the new current differs from old."""
        self.tab_bar.blockSignals(True)
        try:
            while self.tab_bar.count() > 0:
                self.tab_bar.removeTab(0)
            for n in names:
                self.tab_bar.addTab(n)
            if 0 <= current_index < len(names):
                self.tab_bar.setCurrentIndex(current_index)
        finally:
            self.tab_bar.blockSignals(False)

    def set_current_index(self, idx):
        self.tab_bar.blockSignals(True)
        try:
            if 0 <= idx < self.tab_bar.count():
                self.tab_bar.setCurrentIndex(idx)
        finally:
            self.tab_bar.blockSignals(False)

    def current_index(self):
        return self.tab_bar.currentIndex()

    def set_tab_text(self, idx, text):
        """Override the visible text on a single tab without disturbing
        the rest.  Used for transient badges (e.g., 'Optimize • running')."""
        if 0 <= idx < self.tab_bar.count():
            self.tab_bar.setTabText(idx, text)

    def find_tab(self, name):
        """Return the index of the tab with this exact name, or -1."""
        for i in range(self.tab_bar.count()):
            if self.tab_bar.tabText(i) == name:
                return i
        return -1

    def _on_context_menu(self, pos):
        idx = self.tab_bar.tabAt(pos)
        if idx < 0:
            return
        m = QMenu(self)
        a_manage = m.addAction('Manage Docks…')
        a_rename = m.addAction('Rename…')
        a_dup = m.addAction('Duplicate')
        m.addSeparator()
        a_del = m.addAction('Delete')
        act = m.exec(self.tab_bar.mapToGlobal(pos))
        if act is a_manage:
            self.manage_requested.emit(idx)
        elif act is a_rename:
            self.rename_requested.emit(idx)
        elif act is a_dup:
            self.duplicate_requested.emit(idx)
        elif act is a_del:
            self.delete_requested.emit(idx)


# ---------------------------------------------------------------------------
#  Manage-docks dialog
# ---------------------------------------------------------------------------

class PinnedDocksDialog(QDialog):
    """Pick docks that should be visible on every workspace tab."""

    def __init__(self, parent, pinned_set, all_docks):
        super().__init__(parent)
        self.setWindowTitle('Pin Docks Across All Workspaces')
        self.resize(420, 520)

        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(
            'Pinned docks stay visible on every workspace, regardless of\n'
            'whether the active workspace lists them.  Useful for the\n'
            'element table or System Data dock you always want at hand.'))

        self.list = QListWidget()
        for name in sorted(all_docks, key=lambda n: all_docks[n].lower()):
            title = all_docks[name]
            item = QListWidgetItem(f'{title}   ({name})')
            item.setData(Qt.UserRole, name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(
                Qt.Checked if name in pinned_set else Qt.Unchecked)
            self.list.addItem(item)
        lay.addWidget(self.list)

        btns = QHBoxLayout()
        btn_ok = QPushButton('OK')
        btn_cancel = QPushButton('Cancel')
        btn_ok.setDefault(True)
        btns.addStretch(1)
        btns.addWidget(btn_ok)
        btns.addWidget(btn_cancel)
        lay.addLayout(btns)
        btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)

    def selected_dock_names(self):
        out = set()
        for i in range(self.list.count()):
            it = self.list.item(i)
            if it.checkState() == Qt.Checked:
                out.add(it.data(Qt.UserRole))
        return out


class ManageWorkspaceDialog(QDialog):
    """Check which docks belong to a given workspace."""

    def __init__(self, parent, workspace, all_docks):
        """
        workspace  : Workspace instance being edited
        all_docks  : dict { objectName: human-title }
        """
        super().__init__(parent)
        self.setWindowTitle(f'Manage docks — "{workspace.name}"')
        self.resize(420, 520)
        self._workspace = workspace

        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(
            'Check the docks you want visible on this workspace tab:'))

        self.list = QListWidget()
        current = set(workspace.dock_names)
        for name in sorted(all_docks, key=lambda n: all_docks[n].lower()):
            title = all_docks[name]
            item = QListWidgetItem(f'{title}   ({name})')
            item.setData(Qt.UserRole, name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(
                Qt.Checked if name in current else Qt.Unchecked)
            self.list.addItem(item)
        lay.addWidget(self.list)

        btns = QHBoxLayout()
        btn_all = QPushButton('All')
        btn_none = QPushButton('None')
        btn_ok = QPushButton('OK')
        btn_cancel = QPushButton('Cancel')
        btn_ok.setDefault(True)
        btns.addWidget(btn_all)
        btns.addWidget(btn_none)
        btns.addStretch(1)
        btns.addWidget(btn_ok)
        btns.addWidget(btn_cancel)
        lay.addLayout(btns)

        btn_all.clicked.connect(lambda: self._set_all(Qt.Checked))
        btn_none.clicked.connect(lambda: self._set_all(Qt.Unchecked))
        btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)

    def _set_all(self, state):
        for i in range(self.list.count()):
            self.list.item(i).setCheckState(state)

    def selected_dock_names(self):
        out = []
        for i in range(self.list.count()):
            it = self.list.item(i)
            if it.checkState() == Qt.Checked:
                out.append(it.data(Qt.UserRole))
        return out


# ---------------------------------------------------------------------------
#  Default dock groupings
# ---------------------------------------------------------------------------

#  Bumped each release that adds a new default workspace.  Used by
#  load_json() to merge new defaults into existing saved blobs without
#  overwriting user customizations.
DEFAULTS_REVISION = 2

DEFAULT_WORKSPACES = [
    ('Design', [
        'welcome', 'layout', 'layout3d', 'summary', 'library',
    ]),
    ('Optimize', [
        'layout', 'optimizer', 'sliders', 'multiconfig',
        'snapshots', 'summary',
    ]),
    ('Analysis', [
        'layout', 'spot', 'rayfan', 'footprint', 'distortion',
        'spot_field', 'through_focus', 'psfmtf', 'field_browser',
        'summary',
    ]),
    ('Wave Optics', [
        'layout', 'waveoptics', 'zernike', 'interferometry',
        'phase_retrieval', 'ghost',
    ]),
    ('Tolerancing', [
        'layout', 'tolerance', 'sensitivity', 'summary',
    ]),
    ('Materials', [
        'materials', 'glassmap', 'library', 'summary',
    ]),
]


def default_dock_titles():
    """objectName -> human title, used to label entries in Manage Docks."""
    return {
        'layout': '2D Layout',
        'layout3d': '3D Layout',
        'summary': 'System Data',
        'library': 'Library',
        'multiconfig': 'Multi-Config',
        'optimizer': 'Optimizer',
        'sliders': 'Sliders',
        'snapshots': 'Snapshots',
        'spot': 'Spot Diagram',
        'rayfan': 'Ray Fan / OPD',
        'footprint': 'Footprint',
        'distortion': 'Distortion',
        'spot_field': 'Spot vs Field',
        'through_focus': 'Through-focus',
        'psfmtf': 'PSF / MTF',
        'field_browser': 'Field Browser',
        'waveoptics': 'Wave Optics',
        'zernike': 'Zernike',
        'interferometry': 'Interferometry',
        'phase_retrieval': 'Phase Retrieval',
        'jones_pupil': 'Jones Pupil',
        'ghost': 'Ghost Analysis',
        'tolerance': 'Tolerance',
        'sensitivity': 'Sensitivity',
        'materials': 'Materials',
        'glassmap': 'Glass Map',
        'diagnostics': 'Diagnostics',
        'welcome': 'Welcome',
        'repl': 'Python',
    }


# ---------------------------------------------------------------------------
#  Manager
# ---------------------------------------------------------------------------

class WorkspaceManager(QObject):
    """Owns the list of workspaces and applies/saves QMainWindow layouts."""

    changed = Signal()

    def __init__(self, main_window, dock_registry):
        """
        main_window    : QMainWindow owning the docks
        dock_registry  : dict { objectName: QDockWidget }
        """
        super().__init__(main_window)
        self.mw = main_window
        self.dock_registry = dict(dock_registry)
        self.workspaces = []
        self.current_index = 0
        # "Pinned" docks are visible on every workspace regardless of
        # the active workspace's dock_names list.  Useful for the
        # element table / system summary that the user always wants.
        self.pinned_docks = set()
        self._suspend_save = False   # raised during programmatic apply

    # ------------------------------------------------------------------
    #  Construction / defaults
    # ------------------------------------------------------------------

    def init_defaults(self):
        """Reset to the built-in default workspace set."""
        self.workspaces = []
        for name, docks in DEFAULT_WORKSPACES:
            # Keep only docks that actually exist on this window.
            docks = [d for d in docks if d in self.dock_registry]
            self.workspaces.append(Workspace(name, docks))
        self.current_index = 0

    def titles(self):
        return [w.name for w in self.workspaces]

    def current(self):
        if 0 <= self.current_index < len(self.workspaces):
            return self.workspaces[self.current_index]
        return None

    # ------------------------------------------------------------------
    #  Apply / save layout
    # ------------------------------------------------------------------

    def apply_index(self, idx):
        """Show the workspace at idx.  Does not save the outgoing one;
        callers should call save_current_layout() first when switching."""
        if not (0 <= idx < len(self.workspaces)):
            return
        ws = self.workspaces[idx]
        self.current_index = idx
        self._suspend_save = True
        try:
            wanted = set(ws.dock_names) | set(self.pinned_docks)
            # Restore saved dock geometry first if we have one -- that
            # sets positions AND visibility according to the blob.
            if ws.state_b64:
                try:
                    blob = QByteArray(base64.b64decode(ws.state_b64))
                    self.mw.restoreState(blob)
                except Exception:
                    pass
            # Enforce visibility explicitly: membership is the source of
            # truth, not whatever was inside the blob.
            for name, dock in self.dock_registry.items():
                dock.setVisible(name in wanted)
        finally:
            self._suspend_save = False
        self.changed.emit()

    # ------------------------------------------------------------------
    #  Pinned docks (visible on every workspace)
    # ------------------------------------------------------------------

    def pin(self, dock_name):
        if dock_name in self.dock_registry:
            self.pinned_docks.add(dock_name)

    def unpin(self, dock_name):
        self.pinned_docks.discard(dock_name)

    def is_pinned(self, dock_name):
        return dock_name in self.pinned_docks

    def save_current_layout(self):
        """Save the QMainWindow dock layout into the current workspace."""
        if self._suspend_save:
            return
        ws = self.current()
        if ws is None:
            return
        try:
            blob = bytes(self.mw.saveState().data())
            ws.state_b64 = base64.b64encode(blob).decode('ascii')
        except Exception:
            pass

    # ------------------------------------------------------------------
    #  Visibility hooks (wired by MainWindow)
    # ------------------------------------------------------------------

    def on_user_toggled_dock(self, dock_name, visible):
        """Called when the user explicitly showed/hid a dock via the
        View menu or close button.  Updates the current workspace's
        dock_names so the change is persistent for that workspace."""
        if self._suspend_save:
            return
        ws = self.current()
        if ws is None:
            return
        present = dock_name in ws.dock_names
        if visible and not present:
            ws.dock_names.append(dock_name)
        elif not visible and present:
            ws.dock_names.remove(dock_name)

    # ------------------------------------------------------------------
    #  CRUD
    # ------------------------------------------------------------------

    def add(self, name=None, dock_names=None, state_b64=None):
        if not name:
            name = self._unique_name('New Workspace')
        else:
            name = self._unique_name(name)
        self.workspaces.append(Workspace(name, dock_names or [], state_b64))
        return len(self.workspaces) - 1

    def duplicate(self, idx):
        if not (0 <= idx < len(self.workspaces)):
            return -1
        src = self.workspaces[idx]
        dup = Workspace(
            self._unique_name(src.name + ' copy'),
            list(src.dock_names),
            src.state_b64,
        )
        self.workspaces.insert(idx + 1, dup)
        return idx + 1

    def remove(self, idx):
        """Remove a workspace.  Refuses to delete the last one."""
        if not (0 <= idx < len(self.workspaces)):
            return False
        if len(self.workspaces) <= 1:
            return False
        self.workspaces.pop(idx)
        if self.current_index >= len(self.workspaces):
            self.current_index = len(self.workspaces) - 1
        return True

    def rename(self, idx, new_name):
        if 0 <= idx < len(self.workspaces) and new_name.strip():
            self.workspaces[idx].name = new_name.strip()

    def set_dock_names(self, idx, names):
        if 0 <= idx < len(self.workspaces):
            self.workspaces[idx].dock_names = list(names)

    def _unique_name(self, base):
        existing = {w.name for w in self.workspaces}
        if base not in existing:
            return base
        i = 2
        while f'{base} {i}' in existing:
            i += 1
        return f'{base} {i}'

    # ------------------------------------------------------------------
    #  Persistence
    # ------------------------------------------------------------------

    def to_json(self):
        return json.dumps({
            'current': int(self.current_index),
            'pinned': sorted(self.pinned_docks),
            'defaults_revision': DEFAULTS_REVISION,
            'workspaces': [w.to_dict() for w in self.workspaces],
        })

    def load_json(self, s):
        """Replace workspaces from a JSON blob.  Returns True on success.
        If the blob predates the current DEFAULTS_REVISION, missing
        default workspaces are appended so users get new tabs (e.g.
        'Optimize' in 3.2.12) without losing their customizations."""
        try:
            d = json.loads(s) if isinstance(s, str) else s
            ws_list = [Workspace.from_dict(x)
                       for x in (d.get('workspaces') or [])]
            if not ws_list:
                return False
            self.workspaces = ws_list
            self.pinned_docks = {
                x for x in (d.get('pinned') or [])
                if x in self.dock_registry
            }
            self.current_index = max(
                0, min(len(self.workspaces) - 1, int(d.get('current', 0))))

            saved_rev = int(d.get('defaults_revision', 0))
            if saved_rev < DEFAULTS_REVISION:
                existing_names = {w.name for w in self.workspaces}
                for name, docks in DEFAULT_WORKSPACES:
                    if name not in existing_names:
                        docks = [x for x in docks
                                 if x in self.dock_registry]
                        self.workspaces.append(Workspace(name, docks))
            return True
        except Exception:
            return False
