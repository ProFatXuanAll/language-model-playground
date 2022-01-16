"""Language model module.

Attributes
----------
ALL_MODELS: List[:py:class:`lmp.model.BaseModel`]
  All available language models.
MODEL_OPTS: Final[Dict[str, :py:class:`lmp.model.BaseModel`]]
  Mapping from language model's name `model_name` to language model's class.

Examples
--------
Get :py:class:`lmp.model.RNNModel` by its name.

>>> from lmp.model import MODEL_OPTS, RNNModel
>>> RNNModel.model_name in MODEL_OPTS
True
>>> MODEL_OPTS[RNNModel.model_name] == RNNModel
True
"""

from typing import Dict, Final, List, Type

from lmp.model._base import BaseModel
from lmp.model._gru import GRUModel
from lmp.model._lstm import LSTMModel
from lmp.model._rnn import RNNModel

ALL_MODELS: Final[List[Type[BaseModel]]] = [
  GRUModel,
  LSTMModel,
  RNNModel,
]
MODEL_OPTS: Final[Dict[str, Type[BaseModel]]] = {m.model_name: m for m in ALL_MODELS}
