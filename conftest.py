"""Collection rules shared by the whole suite.

FastAPI comes with the optional ``api`` extra. Without it, the API tests and the app module
(collected for its doctests) cannot even be imported, and an import error at collection stops
the run even when a marker such as ``-m media`` would have deselected every one of them. So a
job that installs only another extra leaves them out instead of failing on them.
"""

from __future__ import annotations

import importlib.util

collect_ignore: list[str] = []

if importlib.util.find_spec("fastapi") is None:
    collect_ignore += ["tests/api", "src/openwave/api/app.py"]
