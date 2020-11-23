r"""Path variables.

Example
=======

::

    import lmp.path

    data_path = lmp.path.DATA_PATH
    exp_path = lmp.path.EXP_PATH
    log_path = lmp.path.LOG_PATH
"""


import os
from typing import Final

PROJECT_ROOT: Final[str] = os.path.abspath(os.path.join(
    os.path.abspath(__file__),
    os.pardir,
    os.pardir
))
DATA_PATH: Final[str] = os.path.join(PROJECT_ROOT, 'data')
EXP_PATH: Final[str] = os.path.join(PROJECT_ROOT, 'exp')
LOG_PATH: Final[str] = os.path.join(EXP_PATH, 'log')
