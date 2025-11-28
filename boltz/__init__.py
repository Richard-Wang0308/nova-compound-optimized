# Boltz module
# This package includes both boltz/ (with enhanced_wrapper.py) and boltz/boltz/ (with main.py, data/, etc.)
import os
import sys

# Extend the package path to include boltz/boltz/ so imports like "from boltz.main" work
_boltz_boltz_dir = os.path.join(os.path.dirname(__file__), "boltz")
if os.path.exists(_boltz_boltz_dir) and os.path.isdir(_boltz_boltz_dir):
    # Add boltz/boltz to the package path so submodules can be found
    __path__.append(_boltz_boltz_dir)
