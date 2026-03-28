import sys
import os

# Ensure the local workspace copy of mlx_lm takes precedence over the
# system-installed version so all patches are picked up during testing.
sys.path.insert(0, os.path.dirname(__file__))
