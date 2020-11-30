r"""Optimizer module.

All optimizer classes must be re-imported in this file.

Attributes
==========
ALL_OPTIMS: List[:py:class:`lmp.optim.BaseOptim`]
    All available optimizers.
    Every time a new optimizer is added, it must also be added to
    ``ALL_OPTIMS`` list.
OPTIM_OPTS: Final[Dict[str, :py:class:`lmp.optim.BaseOptim`]]
    Mapping from optimizer's name to optimizer's class.
    All optimizers must have class attribute ``optim_name``.

Examples
========
Check ``'SGD'`` is an available optimizer class.

>>> from lmp.optim import OPTIM_OPTS
>>> 'SGD' in OPTIM_OPTS
True

Get ``'SGD'`` optimizer class.

>>> from lmp.optim import SGDOptim
>>> OPTIM_OPTS['SGD'] == SGDOptim
True
"""


from typing import Dict, Final, List

from lmp.optim._base import BaseOptim
from lmp.optim._sgd import SGDOptim

ALL_OPTIMS: Final[List[BaseOptim]] = [
    SGDOptim,
]
OPTIM_OPTS: Final[Dict[str, BaseOptim]] = {t.optim_name: t for t in ALL_OPTIMS}
