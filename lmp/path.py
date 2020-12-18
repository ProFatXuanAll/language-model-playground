r"""Path variables shared throughout this project.

Attributes
==========
PROJECT_ROOT: Final[str]
    Absolute path of the project root directory.
DATA_PATH: Final[str]
    Absolute path of all the dataset.
    Some of the dataset size are too big to be tracked on GitHub, thus we host
    all of our dataset files on different GitHub repository.
    See :py:class:`lmp.dset.BaseDset` for more information.
EXP_PATH: Final[str]
    Absolute path of all experiments.
    Experiment are ignore by `.git`, no experiment results (model checkpoints,
    tokenizer cofiguration, etc.) will be commited.
LOG_PATH: Final[str]
    Absolute path of all experiments' log.
    Experiment are ignore by `.git`, no experiment logs will be commited.

Example
=======
>>> import lmp.path
>>> isinstance(lmp.path.PROJECT_ROOT, str)
True
>>> isinstance(lmp.path.DATA_PATH, str)
True
>>> isinstance(lmp.path.EXP_PATH, str)
True
>>> isinstance(lmp.path.LOG_PATH, str)
True
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
