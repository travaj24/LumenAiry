#!/usr/bin/env python
"""
Launch the Optical Designer application.

Usage
-----
::

    python run_optical_designer.py
    python run_optical_designer.py --demo            # load a demo lens
    python run_optical_designer.py path/to/file.zmx  # open a .zmx file
    python run_optical_designer.py path/to/file.txt  # open a prescription text file

Author: Andrew Traverso
"""

import sys


def main():
    from PySide6.QtWidgets import QApplication
    from lumenairy.ui.main_window import MainWindow, apply_dark_theme

    app = QApplication(sys.argv)
    app.setApplicationName('Optical Designer')


    apply_dark_theme(app)

    window = MainWindow()

    # Handle command-line arguments
    args = sys.argv[1:]
    if '--demo' in args:
        from lumenairy.prescriptions import thorlabs_lens
        rx = thorlabs_lens('AC254-100-C')
        window.model.load_prescription(rx, 1310.0)
    elif args and not args[0].startswith('-'):
        filepath = args[0]
        try:
            if filepath.lower().endswith('.zmx'):
                from lumenairy.prescriptions import load_zmx_prescription
                rx = load_zmx_prescription(filepath)
            else:
                from lumenairy.prescriptions import load_zemax_prescription_txt
                rx = load_zemax_prescription_txt(filepath)
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
