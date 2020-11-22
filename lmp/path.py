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

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.abspath(__file__),
    os.pardir,
    os.pardir
))
DATA_PATH = os.path.join(
    PROJECT_ROOT,
    'data'
)
EXP_PATH = os.path.join(
    PROJECT_ROOT,
    'exp'
)
LOG_PATH = os.path.join(
    EXP_PATH,
    'log'
)
