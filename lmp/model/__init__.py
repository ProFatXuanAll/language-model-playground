"""Language model module.

Attributes
----------
ALL_MODELS: list[lmp.model.BaseModel]
  All available language models.
MODEL_OPTS: typing.Final[dict[str, lmp.model.BaseModel]]
  Mapping from language model's name `model_name` to language model's class.

Examples
--------
Get :py:class:`lmp.model.ElmanNet` by its name.

>>> from lmp.model import MODEL_OPTS, ElmanNet
>>> ElmanNet.model_name in MODEL_OPTS
True
>>> MODEL_OPTS[ElmanNet.model_name] == ElmanNet
True
"""

from typing import Dict, Final, List, Type

from lmp.model._base import BaseModel
from lmp.model._elman_net import ElmanNet

ALL_MODELS: Final[List[Type[BaseModel]]] = [
  ElmanNet,
]
MODEL_OPTS: Final[Dict[str, Type[BaseModel]]] = {m.model_name: m for m in ALL_MODELS}
