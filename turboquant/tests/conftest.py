"""
pytest conftest for turboquant/ tests.
Ensures the project root is on sys.path so ``import turboquant`` works.
"""
import os
import sys

# Project root is two levels up from this file (turboquant/tests/conftest.py)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
