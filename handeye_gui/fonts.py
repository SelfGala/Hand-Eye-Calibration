import os
import sys
from pathlib import Path
from PyQt5.QtGui import QFont, QFontDatabase


def configure_fonts(app):
    # Windows' offscreen Qt plugin does not enumerate installed system fonts.
    # Load the local CJK font for both readable UI text and truthful QA screenshots.
    database = QFontDatabase()
    if sys.platform == "win32" and not database.families():
        path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "msyh.ttc"
        if path.exists():
            QFontDatabase.addApplicationFont(str(path))
    families = set(QFontDatabase().families())
    for family in ("Microsoft YaHei UI", "Noto Sans CJK SC", "WenQuanYi Micro Hei", "DejaVu Sans"):
        if family in families:
            app.setFont(QFont(family, 10))
            break

