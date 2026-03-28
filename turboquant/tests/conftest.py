"""
Compatibility stub.  Unit tests have moved to tests/unit/.

This conftest is retained so that ``pytest turboquant/tests/`` still adds the
project root to sys.path (needed if someone runs from that directory directly).
Canonical test command: ``pytest tests/``
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
