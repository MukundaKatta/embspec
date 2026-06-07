"""Test package bootstrap.

embspec uses a ``src/`` layout, so ``embspec`` is only importable after the
package is installed (as CI does via ``uv``/pytest) *or* with ``src`` on the
import path. To keep the stdlib runner working without an install step —

    python3 -m unittest discover -s tests

we prepend the repo's ``src`` directory to ``sys.path`` here. This is a no-op
when the package is already importable (e.g. an editable/CI install), so it
does not interfere with the pytest-based CI run.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    try:
        import embspec  # noqa: F401  # already importable (installed) -> skip
    except ImportError:
        sys.path.insert(0, str(_SRC))
