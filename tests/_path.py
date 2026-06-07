"""Make ``embspec`` importable from a ``src/`` layout without an install step.

Imported first by every test module so the suite runs under the stdlib runner
regardless of how discovery is invoked::

    python3 -m unittest discover -s tests
    python3 -m unittest discover            # tests as a package

It is a no-op when ``embspec`` is already importable (e.g. the editable/CI
install used by the pytest job), so it never shadows an installed package.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    try:
        import embspec  # noqa: F401
    except ImportError:
        src = Path(__file__).resolve().parent.parent / "src"
        if src.is_dir() and str(src) not in sys.path:
            sys.path.insert(0, str(src))


_ensure_src_on_path()
