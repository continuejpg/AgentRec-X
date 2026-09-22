"""Make ``src/`` importable for the backend's own tests.

Kept local to the backend so AgentRec-X's suite - which never enters this directory, because
``pytest.ini`` sets ``norecursedirs = backends`` - needs no path manipulation and no backend
dependency.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
