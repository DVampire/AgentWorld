#!/usr/bin/env python3
"""SkillsBench sandbox server entrypoint (CLI args from terminal)."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    # Supports: python3 sandbox/start_sandbox_server.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from sandbox.cli import main
else:
    # Supports: python3 -m sandbox.start_sandbox_server
    from .cli import main


if __name__ == "__main__":
    main()
