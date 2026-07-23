"""
Backward-compat shim: assembly is now the general, SUBJECT-driven
assemble_container.py (it detects the glass shell by name/material rather than
assuming the fixed 'glass_dome'/'rock_base'/'penguins' snowglobe names).

Kept so `make assemble-snowglobe` and existing invocations keep working. New
work should call assemble_container.py.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import assemble_container  # noqa: E402

if __name__ == '__main__':
    assemble_container.main()
