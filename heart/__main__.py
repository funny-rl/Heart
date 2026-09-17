"""`python -m heart` opens the human play server.

`heart/__init__.py` imports `heart.play`, so running that submodule directly
would execute it a second time as `__main__`; this entry point avoids that.
"""

from __future__ import annotations

from heart.play import main

if __name__ == "__main__":
    main()
