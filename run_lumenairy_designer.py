#!/usr/bin/env python
"""
Launch the LumenAiry Designer application.

Usage
-----
::

    python run_lumenairy_designer.py
    python run_lumenairy_designer.py --demo            # load a demo lens
    python run_lumenairy_designer.py path/to/file.zmx  # open a .zmx file
    python run_lumenairy_designer.py path/to/file.txt  # open a prescription text file

Author: Andrew Traverso
"""

import sys


def main():
    from PySide6.QtWidgets import QApplication

    from lumenairy.ui.main_window import MainWindow, apply_dark_theme

    app = QApplication(sys.argv)
    app.setApplicationName('LumenAiry Designer')

    apply_dark_theme(app)

    window = MainWindow()

    # Handle command-line arguments.
    # Prescription helpers (thorlabs_lens, load_zemax_zmx,
    # load_zemax_prescription_data_txt) are exported at the top-level
    # `lumenairy` namespace; the original launcher tried to import
    # them from a non-existent `lumenairy.prescriptions` submodule,
    # which the IDE rightly flagged but Python silently masked
    # because the --demo branch was rarely the one users hit.
    import lumenairy as la
    args = sys.argv[1:]
    if '--demo' in args:
        rx = la.thorlabs_lens('AC254-100-C')
        window.model.load_prescription(rx, 1310.0)
    elif args and not args[0].startswith('-'):
        filepath = args[0]
        try:
            if filepath.lower().endswith('.zmx'):
                rx = la.load_zemax_zmx(filepath)
            else:
                rx = la.load_zemax_prescription_data_txt(filepath)
            wv_nm = rx.get('wavelength', 1310e-9)
            if isinstance(wv_nm, float) and wv_nm < 1e-3:
                wv_nm = wv_nm * 1e9
            window.model.load_prescription(rx, wv_nm)
        except Exception as e:
            print(f'Error loading {filepath}: {e}', file=sys.stderr)

    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
