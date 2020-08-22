r"""Path variables.

Usages:
    import lmp.path

    data_path = lmp.path.DATA_PATH
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.abspath(__file__),
    os.pardir,
    os.pardir
))
DATA_PATH = os.path.join(
    PROJECT_ROOT,
    'data'
)
