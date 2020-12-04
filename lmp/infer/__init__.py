r"""Language model inference methods module.

All language model inference method classes must be re-imported in this file.

Attributes
==========
ALL_INFERS: List[:py:class:`lmp.model.BaseInfer`]
    All available language model inference methods.
    Every time a new inference method is added, it must also be added to
    ``ALL_INFERS`` list.
INFER_OPTS: Final[Dict[str, :py:class:`lmp.model.BaseInfer`]]
    Mapping from inference method's name to inference method's class.
    All inference methods must have class attribute ``infer_name``.

Examples
========
Check ``'top-1'`` is an available inference method.

>>> from lmp.infer import INFER_OPTS
>>> 'top-1' in INFER_OPTS
True

Get ``'top-1'`` inference method class.

>>> from lmp.infer import Top1Model
>>> INFER_OPTS['top-1'] == Top1Model
True
"""


from typing import Dict, Final, List

from lmp.infer._base import BaseInfer
from lmp.infer._top_1 import Top1Infer

ALL_INFERS: Final[List[BaseInfer]] = [
    Top1Infer,
]
INFER_OPTS: Final[Dict[str, BaseInfer]] = {i.infer_name: i for i in ALL_INFERS}
