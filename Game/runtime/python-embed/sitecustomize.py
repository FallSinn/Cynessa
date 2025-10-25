"""Ensure embedded Python can locate bundled packages and game code."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SITE_PACKAGES = ROOT / "runtime" / "site-packages"
GAME_DIR = ROOT / "game"

if SITE_PACKAGES.exists():
    sys.path.insert(0, SITE_PACKAGES.as_posix())
if GAME_DIR.exists():
    sys.path.insert(0, GAME_DIR.as_posix())

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
