"""
Pytest configuration: makes sure the project root (where config.py,
pipeline.py, etc. live) is importable regardless of where `pytest` is
invoked from.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
