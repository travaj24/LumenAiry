"""
Algebra dock -- symbolic optical-operator chain editor.

Surfaces the v4.15.1+ ``lumenairy.algebra`` operator family
(FreeSpace, ThinLens, CylindricalLens, Magnify, FourierTransform,
Aperture, GaussianAperture) as an interactive chain editor.

Per v5.4 GUI-vs-library audit Part 6.2 item P2-A: the operator
algebra ships as a library-only feature in v4.15.x / v5.3.x with no
GUI surface.  This dock closes that gap with chain editing, ABCD
inspection, field application, and JSON save/load.

The tree shows operators in APPLICATION order (top = applied first),
matching ``CompositeOperator._chain``.  Distances enter as mm in the
GUI; operator constructors always receive metres.

Author: Andrew Traverso
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

try:
    from matplotlib.backends.backend_qtagg import (
        FigureCanvasQTAgg as FigureCanvas,
    )
    from matplotlib.figure import Figure
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# Unit conversion factors (GUI uses mm / um; library uses m).
_M_TO_MM = 1.0e3
_MM_TO_M = 1.0e-3
_M_TO_UM = 1.0e6
_UM_TO_M = 1.0e-6

OPERATOR_KINDS: List[str] = [
    'FreeSpace',
    'ThinLens',
    'CylindricalLens',
    'Magnify',
    'FourierTransform',
    'Aperture',
    'GaussianAperture',
]

# Propagator backends accepted by FreeSpace.method.
_METHOD_CHOICES = ['auto', 'asm', 'fresnel', 'fraunhofer', 'rs', 'sas']

# Shapes accepted by Aperture.
_APERTURE_SHAPES = ['circular', 'rectangular', 'annular']


# ---------------------------------------------------------------------------
# Parameter dialog
# ---------------------------------------------------------------------------


class _OperatorParamDialog(QDialog):
    """Prompt for operator constructor kwargs; returns SI-unit dict."""

    def __init__(self, kind: str, parent: Optional[QWidget] = None,
                 initial: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f'Add {kind}')
        self.kind = kind
        self._widgets: Dict[str, QWidget] = {}
        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)
        initial = initial or {}
        self._build_form(form, initial)
        # OK / Cancel
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _add_spin(self, form: QFormLayout, label: str, key: str,
                   value: float, suffix: str = '', lo: float = -1.0e6,
                   hi: float = 1.0e6, dec: int = 4) -> None:
        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setDecimals(dec)
        if suffix:
            spin.setSuffix(' ' + suffix)
        spin.setValue(float(value))
        form.addRow(label, spin)
        self._widgets[key] = spin

    def _add_choice(self, form: QFormLayout, label: str, key: str,
                     choices: List[str], default: str) -> None:
        cb = QComboBox()
        cb.addItems(choices)
        idx = choices.index(default) if default in choices else 0
        cb.setCurrentIndex(idx)
        form.addRow(label, cb)
        self._widgets[key] = cb

    def _build_form(self, form: QFormLayout,
                     initial: Dict[str, Any]) -> None:
        kind = self.kind

        def mm(default_m: float) -> float:
            return float(default_m) * _M_TO_MM if np.isfinite(default_m) else 0.0

        if kind == 'FreeSpace':
            self._add_spin(form, 'distance', 'distance',
                           mm(initial.get('distance', 100.0e-3)),
                           suffix='mm', lo=-10000.0, hi=10000.0)
            self._add_choice(form, 'method', 'method', _METHOD_CHOICES,
                             initial.get('method', 'auto'))
        elif kind == 'ThinLens':
            self._add_spin(form, 'focal length f', 'f',
                           mm(initial.get('f', 100.0e-3)),
                           suffix='mm', lo=-10000.0, hi=10000.0)
        elif kind == 'CylindricalLens':
            self._add_spin(form, 'f_x (0 = flat)', 'f_x',
                           mm(initial.get('f_x', float('inf'))),
                           suffix='mm')
            self._add_spin(form, 'f_y (0 = flat)', 'f_y',
                           mm(initial.get('f_y', float('inf'))),
                           suffix='mm')
            note = QLabel('(set to 0 for flat / infinite focal length)')
            note.setStyleSheet('color: #7a94b8; font-size: 10px;')
            form.addRow(note)
        elif kind == 'Magnify':
            self._add_spin(form, 'magnification a_x', 'a_x',
                           float(initial.get('a_x', 1.0)),
                           lo=1.0e-6, hi=1.0e6)
            self._add_spin(form, 'a_y (0 = same as a_x)', 'a_y',
                           float(initial.get('a_y', 0.0) or 0.0),
                           lo=0.0, hi=1.0e6)
        elif kind == 'FourierTransform':
            self._add_spin(form, 'focal length f_focal', 'f_focal',
                           mm(initial.get('f_focal', 100.0e-3)),
                           suffix='mm', lo=1.0e-6, hi=10000.0)
        elif kind == 'Aperture':
            self._add_spin(form, 'diameter', 'diameter',
                           mm(initial.get('diameter', 10.0e-3)),
                           suffix='mm', lo=1.0e-6, hi=10000.0)
            self._add_choice(form, 'shape', 'shape', _APERTURE_SHAPES,
                             initial.get('shape', 'circular'))
            self._add_spin(form, 'inner diameter (annular, 0=auto)',
                           'inner_diameter',
                           mm(initial.get('inner_diameter', 0.0)),
                           suffix='mm', lo=0.0, hi=10000.0)
        elif kind == 'GaussianAperture':
            sigma_um = float(initial.get('sigma', 1.0e-3)) * _M_TO_UM
            self._add_spin(form, 'sigma', 'sigma', sigma_um,
                           suffix='um', lo=1.0e-3, hi=1.0e6, dec=3)
        else:
            form.addRow(QLabel(f'Unknown operator kind: {kind}'))

    def values_si(self) -> Dict[str, Any]:
        """Read widget values and convert to SI units."""
        kind = self.kind
        v: Dict[str, Any] = {}
        if kind == 'FreeSpace':
            v['distance'] = self._widgets['distance'].value() * _MM_TO_M
            v['method'] = self._widgets['method'].currentText()
        elif kind == 'ThinLens':
            v['f'] = self._widgets['f'].value() * _MM_TO_M
        elif kind == 'CylindricalLens':
            fx_mm = self._widgets['f_x'].value()
            fy_mm = self._widgets['f_y'].value()
            v['f_x'] = float('inf') if fx_mm == 0.0 else fx_mm * _MM_TO_M
            v['f_y'] = float('inf') if fy_mm == 0.0 else fy_mm * _MM_TO_M
        elif kind == 'Magnify':
            v['a_x'] = self._widgets['a_x'].value()
            ay = self._widgets['a_y'].value()
            v['a_y'] = None if ay == 0.0 else ay
        elif kind == 'FourierTransform':
            v['f_focal'] = self._widgets['f_focal'].value() * _MM_TO_M
        elif kind == 'Aperture':
            v['diameter'] = self._widgets['diameter'].value() * _MM_TO_M
            v['shape'] = self._widgets['shape'].currentText()
            v['inner_diameter'] = (
                self._widgets['inner_diameter'].value() * _MM_TO_M)
        elif kind == 'GaussianAperture':
            v['sigma'] = self._widgets['sigma'].value() * _UM_TO_M
        return v


# ---------------------------------------------------------------------------
# AlgebraDock
# ---------------------------------------------------------------------------


class AlgebraDock(QWidget):
    """Interactive editor for symbolic optical-operator chains.

    Backed by ``lumenairy.algebra``; composes operators, inspects the
    system ABCD, and applies the chain to the Designer's current
    field.
    """

    def __init__(self, model: Any, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.model = model
        # Chain entries: {'kind': str, 'params': dict-of-SI-kwargs}.
        # Order matches the tree (top = applied first).
        self._chain: List[Dict[str, Any]] = []
        self._last_output: Optional[Tuple[np.ndarray, float]] = None
        self._build_ui()

    # -----------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # ---- Chain editor box -----------------------------------------
        chain_box = QGroupBox('Operator chain (top = applied first)')
        chain_layout = QVBoxLayout(chain_box)

        # Add / remove / reorder controls.
        top_row = QHBoxLayout()
        top_row.addWidget(QLabel('Add:'))
        self.combo_add = QComboBox()
        self.combo_add.addItems(OPERATOR_KINDS)
        top_row.addWidget(self.combo_add)
        self.btn_add = QPushButton('+ Add')
        self.btn_add.clicked.connect(self._on_add_clicked)
        top_row.addWidget(self.btn_add)

        self.btn_remove = QPushButton('- Remove')
        self.btn_remove.clicked.connect(self._remove_selected)
        top_row.addWidget(self.btn_remove)

        self.btn_up = QPushButton('Up')
        self.btn_up.clicked.connect(self._move_up)
        top_row.addWidget(self.btn_up)

        self.btn_down = QPushButton('Down')
        self.btn_down.clicked.connect(self._move_down)
        top_row.addWidget(self.btn_down)

        self.btn_clear = QPushButton('Clear')
        self.btn_clear.clicked.connect(self._clear_chain)
        top_row.addWidget(self.btn_clear)
        top_row.addStretch()
        chain_layout.addLayout(top_row)

        # Tree of operators.
        self.tree = QTreeWidget()
        self.tree.setColumnCount(3)
        self.tree.setHeaderLabels(['kind', 'parameters', 'ABCD'])
        self.tree.setRootIsDecorated(False)
        self.tree.setColumnWidth(0, 140)
        self.tree.setColumnWidth(1, 320)
        self.tree.setColumnWidth(2, 80)
        chain_layout.addWidget(self.tree)

        layout.addWidget(chain_box, stretch=1)

        # ---- Action row -----------------------------------------------
        actions_box = QGroupBox('Actions')
        actions_layout = QHBoxLayout(actions_box)

        self.btn_from_pres = QPushButton('From prescription')
        self.btn_from_pres.setToolTip(
            'Build an operator chain from the current system '
            'prescription via CompositeOperator.from_prescription.')
        self.btn_from_pres.clicked.connect(self._from_prescription)
        actions_layout.addWidget(self.btn_from_pres)

        self.btn_show_abcd = QPushButton('Show full ABCD')
        self.btn_show_abcd.clicked.connect(self._show_abcd)
        actions_layout.addWidget(self.btn_show_abcd)

        self.btn_apply = QPushButton('Apply to current field')
        self.btn_apply.setObjectName('run_button')
        self.btn_apply.clicked.connect(self._apply_chain)
        actions_layout.addWidget(self.btn_apply)

        self.btn_save = QPushButton('Save chain...')
        self.btn_save.clicked.connect(self._save_chain)
        actions_layout.addWidget(self.btn_save)

        self.btn_load = QPushButton('Load chain...')
        self.btn_load.clicked.connect(self._load_chain)
        actions_layout.addWidget(self.btn_load)

        actions_layout.addStretch()
        layout.addWidget(actions_box)

        # ---- Field / ABCD controls ------------------------------------
        cfg_box = QGroupBox('Field & ABCD readout')
        cfg_layout = QHBoxLayout(cfg_box)
        cfg_layout.addWidget(QLabel('grid N:'))
        self.spin_N = QSpinBox()
        self.spin_N.setRange(32, 8192)
        self.spin_N.setValue(256)
        cfg_layout.addWidget(self.spin_N)
        cfg_layout.addWidget(QLabel('dx (um):'))
        self.spin_dx = QDoubleSpinBox()
        self.spin_dx.setRange(0.001, 1000.0)
        self.spin_dx.setDecimals(3)
        self.spin_dx.setValue(4.0)
        cfg_layout.addWidget(self.spin_dx)
        cfg_layout.addStretch()
        layout.addWidget(cfg_box)

        # ---- ABCD display label ---------------------------------------
        self.lbl_abcd = QLabel(
            'ABCD: (empty chain -- identity)')
        self.lbl_abcd.setFont(QFont('Consolas', 10))
        self.lbl_abcd.setStyleSheet(
            'QLabel{color:#5cb8ff;background:#0a0c10;padding:6px;'
            'border:1px solid #1a2030;}')
        self.lbl_abcd.setTextInteractionFlags(
            Qt.TextSelectableByMouse)
        layout.addWidget(self.lbl_abcd)

        # ---- Output intensity canvas ----------------------------------
        if HAS_MPL:
            self.fig = Figure(figsize=(6, 3), dpi=100, tight_layout=True)
            self.canvas = FigureCanvas(self.fig)
            layout.addWidget(self.canvas, stretch=1)
        else:
            layout.addWidget(QLabel('(matplotlib not available)'))
            self.fig = None
            self.canvas = None

        # ---- Status / log ---------------------------------------------
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(110)
        self.log.setFont(QFont('Consolas', 10))
        self.log.setStyleSheet(
            'QTextEdit{background:#0a0c10;color:#7a94b8;border:none}')
        layout.addWidget(self.log)

        self._refresh_tree()

    # -----------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------

    def _log(self, msg: str) -> None:
        self.log.append(msg)

    # -----------------------------------------------------------------
    # Chain mutation
    # -----------------------------------------------------------------

    def _on_add_clicked(self) -> None:
        kind = self.combo_add.currentText()
        self._add_operator(kind)

    def _add_operator(self, kind: str) -> None:
        """Prompt the user for kwargs and append an operator."""
        if kind not in OPERATOR_KINDS:
            self._log(f'Unknown operator kind: {kind}')
            return
        dlg = _OperatorParamDialog(kind, parent=self)
        if dlg.exec() != QDialog.Accepted:
            return
        try:
            params = dlg.values_si()
            # Validate by instantiating once.
            _ = self._instantiate(kind, params)
        except Exception as e:
            QMessageBox.warning(
                self, 'Invalid parameters',
                f'{type(e).__name__}: {e}')
            return
        self._chain.append({'kind': kind, 'params': params})
        self._refresh_tree()
        self._log(f'Added {kind} ({_format_params(kind, params)})')

    def _remove_selected(self) -> None:
        idx = self._selected_row()
        if idx is None:
            self._log('No operator selected.')
            return
        kind = self._chain[idx]['kind']
        del self._chain[idx]
        self._refresh_tree()
        self._log(f'Removed {kind} at row {idx}')

    def _move_up(self) -> None:
        idx = self._selected_row()
        if idx is None or idx == 0:
            return
        self._chain[idx - 1], self._chain[idx] = (
            self._chain[idx], self._chain[idx - 1])
        self._refresh_tree()
        self.tree.setCurrentItem(self.tree.topLevelItem(idx - 1))

    def _move_down(self) -> None:
        idx = self._selected_row()
        if idx is None or idx >= len(self._chain) - 1:
            return
        self._chain[idx], self._chain[idx + 1] = (
            self._chain[idx + 1], self._chain[idx])
        self._refresh_tree()
        self.tree.setCurrentItem(self.tree.topLevelItem(idx + 1))

    def _clear_chain(self) -> None:
        self._chain.clear()
        self._refresh_tree()
        self._log('Chain cleared.')

    def _selected_row(self) -> Optional[int]:
        item = self.tree.currentItem()
        if item is None:
            return None
        idx = self.tree.indexOfTopLevelItem(item)
        return idx if 0 <= idx < len(self._chain) else None

    # -----------------------------------------------------------------
    # Tree population
    # -----------------------------------------------------------------

    def _refresh_tree(self) -> None:
        self.tree.clear()
        for i, step in enumerate(self._chain):
            kind = step['kind']
            params = step['params']
            item = QTreeWidgetItem([
                f'{i}: {kind}',
                _format_params(kind, params),
                '',
            ])
            self.tree.addTopLevelItem(item)
            btn = QPushButton('ABCD')
            btn.setMaximumWidth(70)
            btn.clicked.connect(
                lambda _checked=False, idx=i: self._show_step_abcd(idx))
            self.tree.setItemWidget(item, 2, btn)
        self._refresh_abcd_label()

    def _refresh_abcd_label(self) -> None:
        try:
            M = self._composite_abcd()
        except Exception as e:
            self.lbl_abcd.setText(f'ABCD: error -- {type(e).__name__}: {e}')
            return
        a, b = M[0, 0], M[0, 1]
        c, d = M[1, 0], M[1, 1]
        self.lbl_abcd.setText(
            f'System ABCD ({len(self._chain)} ops):\n'
            f'  [ {a:+10.4f}   {b:+10.4f} ]\n'
            f'  [ {c:+10.4f}   {d:+10.4f} ]')

    # -----------------------------------------------------------------
    # Operator instantiation
    # -----------------------------------------------------------------

    def _instantiate(self, kind: str, params: Dict[str, Any]) -> Any:
        """Construct a single operator instance from kind + params."""
        from lumenairy import algebra as la_alg
        cls = getattr(la_alg, kind)
        if kind == 'FreeSpace':
            return cls(distance=params['distance'],
                       method=params.get('method', 'auto'))
        if kind == 'ThinLens':
            return cls(f=params['f'])
        if kind == 'CylindricalLens':
            return cls(f_x=params.get('f_x', float('inf')),
                       f_y=params.get('f_y', float('inf')))
        if kind == 'Magnify':
            return cls(a_x=params['a_x'], a_y=params.get('a_y'))
        if kind == 'FourierTransform':
            return cls(f_focal=params['f_focal'])
        if kind == 'Aperture':
            return cls(diameter=params['diameter'],
                       shape=params.get('shape', 'circular'),
                       inner_diameter=params.get('inner_diameter', 0.0))
        if kind == 'GaussianAperture':
            return cls(sigma=params['sigma'])
        raise ValueError(f'Unknown operator kind: {kind}')

    def _build_composite(self) -> Optional[Any]:
        """Return a CompositeOperator for the current chain (or None)."""
        if not self._chain:
            return None
        from lumenairy.algebra import CompositeOperator
        ops = [self._instantiate(step['kind'], step['params'])
               for step in self._chain]
        return CompositeOperator(ops)

    def _composite_abcd(self) -> np.ndarray:
        """System 2x2 ABCD.  Empty chain -> identity."""
        if not self._chain:
            return np.eye(2)
        comp = self._build_composite()
        try:
            return comp.abcd
        except ValueError:
            return comp.abcd_x  # Anamorphic fallback.

    # -----------------------------------------------------------------
    # From-prescription factory
    # -----------------------------------------------------------------

    def _from_prescription(self) -> None:
        """Populate the chain from the model's prescription."""
        pres = self._get_prescription()
        if pres is None:
            self._log('No prescription available on the model.')
            return
        wl = self._get_wavelength_m()
        try:
            from lumenairy.algebra import Operator as _Op
            comp = _Op.from_prescription(pres, wavelength=wl)
        except Exception as e:
            QMessageBox.warning(
                self, 'from_prescription failed',
                f'{type(e).__name__}: {e}')
            self._log(
                f'from_prescription failed: {type(e).__name__}: {e}')
            return
        # Reverse-engineer step dicts from each stage's attributes.
        # ``from_prescription`` only emits FreeSpace / ThinLens, but
        # the dispatcher below covers all 8 operator kinds in case the
        # library grows.
        new_chain: List[Dict[str, Any]] = []
        for op in comp.stages():
            step = _step_from_operator(op)
            if step is None:
                self._log(
                    f'Skipping unknown stage class {type(op).__name__}.')
                continue
            new_chain.append(step)
        self._chain = new_chain
        self._refresh_tree()
        self._log(
            f'Loaded {len(new_chain)} operators from prescription at '
            f'wavelength {wl*1e9:.1f} nm.')

    # -----------------------------------------------------------------
    # Show ABCD
    # -----------------------------------------------------------------

    def _show_abcd(self) -> None:
        """Pop up a dialog with the per-stage and composite ABCDs."""
        if not self._chain:
            QMessageBox.information(
                self, 'System ABCD',
                'Chain is empty.  System ABCD is the 2x2 identity.')
            return
        lines: List[str] = []
        try:
            from lumenairy.algebra import CompositeOperator
            ops = [self._instantiate(s['kind'], s['params'])
                   for s in self._chain]
            comp = CompositeOperator(ops)
            for i, op in enumerate(ops):
                lines.append(f'Stage {i}: {type(op).__name__}')
                M = _abcd_of(op)
                lines.append(_fmt_matrix(M, indent='    '))
            lines.append('')
            lines.append('Composite system ABCD:')
            try:
                lines.append(_fmt_matrix(comp.abcd, indent='    '))
            except ValueError:
                lines.append('  (anamorphic -- showing per-axis ABCDs)')
                lines.append('  ABCD_x:')
                lines.append(_fmt_matrix(comp.abcd_x, indent='    '))
                lines.append('  ABCD_y:')
                lines.append(_fmt_matrix(comp.abcd_y, indent='    '))
            try:
                efl = float(comp.efl)
                if np.isfinite(efl):
                    lines.append(f'\nEFL = {efl*1e3:.4f} mm')
                else:
                    lines.append('\nEFL = +inf (afocal)')
            except ValueError:
                lines.append('\nEFL undefined (anamorphic)')
        except Exception as e:
            QMessageBox.warning(
                self, 'ABCD error',
                f'{type(e).__name__}: {e}')
            return
        text = '\n'.join(lines)
        dlg = QDialog(self)
        dlg.setWindowTitle('System ABCD')
        v = QVBoxLayout(dlg)
        te = QTextEdit()
        te.setReadOnly(True)
        te.setFont(QFont('Consolas', 10))
        te.setPlainText(text)
        v.addWidget(te)
        bb = QDialogButtonBox(QDialogButtonBox.Ok)
        bb.accepted.connect(dlg.accept)
        v.addWidget(bb)
        dlg.resize(560, 480)
        dlg.exec()

    def _show_step_abcd(self, idx: int) -> None:
        """Show the per-stage ABCD for a single row's button."""
        if not (0 <= idx < len(self._chain)):
            return
        step = self._chain[idx]
        try:
            op = self._instantiate(step['kind'], step['params'])
            M = _abcd_of(op)
        except Exception as e:
            QMessageBox.warning(
                self, 'ABCD error',
                f'{type(e).__name__}: {e}')
            return
        text = (f'{step["kind"]} stage ABCD:\n\n'
                + _fmt_matrix(M, indent='    '))
        QMessageBox.information(self, f'Stage {idx} ABCD', text)

    # -----------------------------------------------------------------
    # Apply chain to current field
    # -----------------------------------------------------------------

    def _get_wavelength_m(self) -> float:
        wl_nm = getattr(self.model, 'wavelength_nm', None)
        if wl_nm is None:
            return 633e-9
        return float(wl_nm) * 1e-9

    def _get_prescription(self) -> Optional[Dict[str, Any]]:
        fn = getattr(self.model, 'to_prescription', None)
        if not callable(fn):
            return None
        try:
            return fn()
        except Exception as e:
            self._log(
                f'to_prescription failed: {type(e).__name__}: {e}')
            return None

    def _get_current_field(self) -> Optional[Tuple[np.ndarray, float, float]]:
        """Build (E, dx, wavelength).  Preference: waveoptics input,
        SourceDefinition.to_source, then plane-wave fallback."""
        wl = self._get_wavelength_m()
        # 1. Latest wave-optics input field.
        mw = self.window()
        wd = getattr(mw, 'waveoptics_widget', None)
        if wd is not None:
            res = getattr(wd, '_last_results', None)
            if res and res.get('planes'):
                p0 = res['planes'][0]
                E = np.asarray(p0['field'], dtype=np.complex128)
                dx = float(p0.get('dx', res.get('dx', 4e-6)))
                self._log('Field source: latest wave-optics input plane.')
                return E, dx, wl
        # 2. SourceDefinition factory.
        N = int(self.spin_N.value())
        dx = float(self.spin_dx.value()) * 1e-6
        src = getattr(self.model, 'source', None)
        if src is not None:
            try:
                src_inst = src.to_source(
                    N=N, dx_m=dx,
                    epd_m=float(getattr(self.model, 'epd_m', 0.0)))
                self._log(
                    f'Field source: model.source ({src.source_type}).')
                return (np.asarray(src_inst.E, dtype=np.complex128),
                        dx, wl)
            except Exception as e:
                self._log(
                    f'to_source failed: {type(e).__name__}: {e}; '
                    f'falling back to plane wave.')
        # 3. Plane wave with EPD clip.
        E = np.ones((N, N), dtype=np.complex128)
        epd = float(getattr(self.model, 'epd_m', 0.0))
        if epd > 0.0:
            x = (np.arange(N) - N / 2) * dx
            X, Y = np.meshgrid(x, x)
            R2 = X * X + Y * Y
            E[R2 > (epd / 2) ** 2] = 0.0
        self._log('Field source: synthetic plane wave + EPD clip.')
        return E, dx, wl

    def _apply_chain(self) -> None:
        if not self._chain:
            self._log('Empty chain -- nothing to apply.')
            return
        bundle = self._get_current_field()
        if bundle is None:
            self._log('Could not build an input field.')
            return
        E_in, dx, wl = bundle
        try:
            comp = self._build_composite()
            # Use the tuple-call entry point so we get the (possibly
            # changed) output pitch back -- Magnify and Fresnel-backed
            # FreeSpace can rescale dx.
            tup_out = comp((E_in, dx, wl))
            E_out, dx_out = tup_out[0], float(tup_out[1])
        except Exception as e:
            QMessageBox.warning(
                self, 'Apply failed',
                f'{type(e).__name__}: {e}')
            self._log(f'Apply failed: {type(e).__name__}: {e}')
            return
        self._last_output = (E_out, dx_out)
        self._redraw_intensity(E_out, dx_out)
        self._log(
            f'Applied chain ({len(self._chain)} ops): '
            f'in {E_in.shape}@{dx*1e6:.3f}um -> '
            f'out {E_out.shape}@{dx_out*1e6:.3f}um  '
            f'(peak |E|^2 = {float(np.max(np.abs(E_out)**2)):.3e})')

    def _redraw_intensity(self, field: np.ndarray, dx: float) -> None:
        if not HAS_MPL or self.fig is None:
            return
        self.fig.clear()
        ax = self.fig.add_subplot(1, 1, 1)
        I = np.abs(np.asarray(field)) ** 2
        if I.ndim == 2:
            extent_mm = (np.array([
                -I.shape[1] / 2, I.shape[1] / 2,
                -I.shape[0] / 2, I.shape[0] / 2]) * dx * 1e3)
            ax.imshow(I, origin='lower', cmap='magma',
                       extent=extent_mm)
            ax.set_xlabel('x (mm)')
            ax.set_ylabel('y (mm)')
        else:
            ax.plot(I, color='#5cb8ff')
            ax.set_xlabel('index')
            ax.set_ylabel('|E|^2')
        ax.set_title(f'|E|^2 after chain  ({field.shape}, '
                      f'dx={dx*1e6:.3f} um)')
        self.canvas.draw()

    # -----------------------------------------------------------------
    # Save / load
    # -----------------------------------------------------------------

    def _save_chain(self) -> None:
        if not self._chain:
            QMessageBox.information(
                self, 'Save chain', 'Chain is empty.')
            return
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save operator chain', os.getcwd(),
            'JSON files (*.json);;All files (*.*)')
        if not path:
            return
        payload = {
            'version': 1,
            'chain': [
                {'kind': step['kind'],
                 'params': _sanitize_params(step['params'])}
                for step in self._chain],
        }
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            QMessageBox.warning(
                self, 'Save failed',
                f'{type(e).__name__}: {e}')
            return
        self._log(f'Saved chain to {path}')

    def _load_chain(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, 'Load operator chain', os.getcwd(),
            'JSON files (*.json);;All files (*.*)')
        if not path:
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                payload = json.load(f)
        except Exception as e:
            QMessageBox.warning(
                self, 'Load failed',
                f'{type(e).__name__}: {e}')
            return
        chain_raw = payload.get('chain') if isinstance(payload, dict) else None
        if not isinstance(chain_raw, list):
            QMessageBox.warning(
                self, 'Load failed',
                'JSON does not contain a "chain" list.')
            return
        new_chain: List[Dict[str, Any]] = []
        for i, entry in enumerate(chain_raw):
            try:
                kind = str(entry['kind'])
                if kind not in OPERATOR_KINDS:
                    self._log(
                        f'Entry {i}: skipping unknown kind {kind!r}.')
                    continue
                params = _desanitize_params(entry.get('params', {}))
                # Validate by instantiation.
                _ = self._instantiate(kind, params)
                new_chain.append({'kind': kind, 'params': params})
            except Exception as e:
                self._log(
                    f'Entry {i}: skipped ({type(e).__name__}: {e}).')
        self._chain = new_chain
        self._refresh_tree()
        self._log(f'Loaded chain from {path} ({len(new_chain)} ops)')


# ---------------------------------------------------------------------------
# Free helpers
# ---------------------------------------------------------------------------


def _format_params(kind: str, params: Dict[str, Any]) -> str:
    """Compact one-line summary of an operator's kwargs (mm / um)."""
    if kind == 'FreeSpace':
        return (f'd={params.get("distance", 0.0)*1e3:.4f} mm  '
                f'method={params.get("method", "auto")}')
    if kind == 'ThinLens':
        f = params.get('f', float('inf'))
        return f'f={f*1e3:.4f} mm' if np.isfinite(f) else 'f=inf'
    if kind == 'CylindricalLens':
        fx = params.get('f_x', float('inf'))
        fy = params.get('f_y', float('inf'))
        sx = f'{fx*1e3:.4f} mm' if np.isfinite(fx) else 'inf'
        sy = f'{fy*1e3:.4f} mm' if np.isfinite(fy) else 'inf'
        return f'f_x={sx}, f_y={sy}'
    if kind == 'Magnify':
        ay = params.get('a_y')
        if ay is None:
            return f'a={params.get("a_x", 1.0):.4f}'
        return (f'a_x={params.get("a_x", 1.0):.4f}, '
                f'a_y={ay:.4f}')
    if kind == 'FourierTransform':
        return f'f_focal={params.get("f_focal", 0.0)*1e3:.4f} mm'
    if kind == 'Aperture':
        s = (f'D={params.get("diameter", 0.0)*1e3:.4f} mm, '
             f'shape={params.get("shape", "circular")}')
        if params.get('shape') == 'annular':
            s += f', D_in={params.get("inner_diameter", 0.0)*1e3:.4f} mm'
        return s
    if kind == 'GaussianAperture':
        return f'sigma={params.get("sigma", 0.0)*1e6:.3f} um'
    return ', '.join(f'{k}={v!r}' for k, v in params.items())


def _fmt_matrix(M: np.ndarray, indent: str = '') -> str:
    """Pretty-print a 2x2 ABCD matrix."""
    a = np.asarray(M, dtype=float)
    return (
        f'{indent}[ {a[0,0]:+12.6f}   {a[0,1]:+12.6f} ]\n'
        f'{indent}[ {a[1,0]:+12.6f}   {a[1,1]:+12.6f} ]')


def _abcd_of(op: Any) -> np.ndarray:
    """Return an ABCD matrix to display for ``op`` (abcd_x for anamorphic)."""
    try:
        return np.asarray(op.abcd)
    except ValueError:
        return np.asarray(op.abcd_x)


def _step_from_operator(op: Any) -> Optional[Dict[str, Any]]:
    """Reverse-engineer a chain step dict from an Operator instance."""
    cls = type(op).__name__
    if cls == 'FreeSpace':
        return {'kind': cls,
                 'params': {'distance': float(op.distance),
                            'method': str(op.method)}}
    if cls == 'ThinLens':
        return {'kind': cls, 'params': {'f': float(op.f)}}
    if cls == 'CylindricalLens':
        return {'kind': cls,
                 'params': {'f_x': float(op.f_x), 'f_y': float(op.f_y)}}
    if cls == 'Magnify':
        return {'kind': cls,
                 'params': {'a_x': float(op.a_x), 'a_y': float(op.a_y)}}
    if cls == 'FourierTransform':
        return {'kind': cls, 'params': {'f_focal': float(op.f)}}
    if cls == 'Aperture':
        return {'kind': cls,
                 'params': {'diameter': float(op.diameter),
                            'shape': str(op.shape),
                            'inner_diameter': float(op.inner_diameter)}}
    if cls == 'GaussianAperture':
        return {'kind': cls, 'params': {'sigma': float(op.sigma)}}
    return None


def _sanitize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """JSON-safe params: replace +/-inf / nan with string tokens."""
    out: Dict[str, Any] = {}
    for k, v in params.items():
        if isinstance(v, float):
            if np.isinf(v):
                out[k] = 'inf' if v > 0 else '-inf'
            elif np.isnan(v):
                out[k] = 'nan'
            else:
                out[k] = float(v)
        else:
            out[k] = v
    return out


def _desanitize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Inverse of :func:`_sanitize_params`."""
    tokens = {'inf': float('inf'), '-inf': float('-inf'),
              'nan': float('nan')}
    return {k: tokens.get(v, v) if isinstance(v, str) else v
            for k, v in params.items()}


__all__ = ['AlgebraDock']
