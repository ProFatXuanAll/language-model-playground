"""Language model inference methods module.

Attributes
----------
ALL_INFERS: List[:py:class:`lmp.model.BaseInfer`]
  All available language model inference methods.
INFER_OPTS: Final[Dict[str, :py:class:`lmp.model.BaseInfer`]]
  Mapping inference method name ``infer_name`` to inference method class.

Examples
--------
Get :py:class:`lmp.infer.Top1Infer` by its name.

>>> from lmp.infer import INFER_OPTS, Top1Infer
>>> Top1Infer.infer_name in INFER_OPTS
True
>>> INFER_OPTS[Top1Infer.infer_name] == Top1Infer
True
"""

from typing import Dict, Final, List, Type

from lmp.infer._base import BaseInfer
from lmp.infer._top_1 import Top1Infer
from lmp.infer._top_k import TopKInfer
from lmp.infer._top_p import TopPInfer

ALL_INFERS: Final[List[Type[BaseInfer]]] = [
  Top1Infer,
  TopKInfer,
  TopPInfer,
]
INFER_OPTS: Final[Dict[str, Type[BaseInfer]]] = {i.infer_name: i for i in ALL_INFERS}
