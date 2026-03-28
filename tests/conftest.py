"""
Root pytest conftest for the unified tests/ tree.

Adds the project root to sys.path so both ``import turboquant`` and
``import mlx_lm`` resolve correctly regardless of where pytest is invoked.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
