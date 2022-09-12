"""Language model module.

Attributes
----------
ALL_MODELS: list[lmp.model.BaseModel]
  All available language models.
MODEL_OPTS: typing.Final[dict[str, lmp.model.BaseModel]]
  Mapping from language model name ``model_name`` to language model class.

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

import torch

from lmp.model._base import BaseModel
from lmp.model._elman_net import ElmanNet, ElmanNetLayer
from lmp.model._lstm_1997 import LSTM1997, LSTM1997Layer
from lmp.model._lstm_2000 import LSTM2000, LSTM2000Layer
from lmp.model._lstm_2002 import LSTM2002, LSTM2002Layer
from lmp.model._trans_enc import MultiHeadAttnLayer, PosEncLayer, TransEnc, TransEncLayer

ALL_MODELS: Final[List[Type[BaseModel]]] = [
  ElmanNet,
  LSTM1997,
  LSTM2000,
  LSTM2002,
  TransEnc,
]
MODEL_OPTS: Final[Dict[str, Type[BaseModel]]] = {m.model_name: m for m in ALL_MODELS}
SUB_MODELS: Final[List[Type[torch.nn.Module]]] = [
  ElmanNetLayer,
  LSTM1997Layer,
  LSTM2000Layer,
  LSTM2002Layer,
  MultiHeadAttnLayer,
  PosEncLayer,
  TransEncLayer,
]
