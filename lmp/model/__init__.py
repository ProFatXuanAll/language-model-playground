r"""Neural network language model module.

All model classes must be re-imported in this file.

Attributes
==========
ALL_MODELS: List[:py:class:`lmp.model.BaseModel`]
    All available models.
    Every time a new model is added, it must also be added to ``ALL_MODELS``
    list.
MODEL_OPTS: Final[Dict[str, :py:class:`lmp.model.BaseModel`]]
    Mapping from model's name to model's class.
    All models must have class attribute ``model_name``.

Examples
========
Check ``'RNN'`` is available model.

>>> from lmp.model import MODEL_OPTS
>>> 'RNN' in MODEL_OPTS
True

Get ``'RNN'`` model class.

>>> from lmp.model import RNNModel
>>> MODEL_OPTS['RNN'] == RNNModel
True
"""


from typing import Dict, Final, List

from lmp.model._attention_mechanism import attention_mechanism
from lmp.model._base import BaseModel
from lmp.model._base_res_rnn_block import BaseResRNNBlock
from lmp.model._base_res_rnn_model import BaseResRNNModel
from lmp.model._base_rnn_model import BaseRNNModel
from lmp.model._base_self_attention_res_rnn_model import \
    BaseSelfAttentionResRNNModel
from lmp.model._base_self_attention_rnn_model import BaseSelfAttentionRNNModel
from lmp.model._gru_model import GRUModel
from lmp.model._lstm_model import LSTMModel
from lmp.model._res_gru_block import ResGRUBlock
from lmp.model._res_gru_model import ResGRUModel
from lmp.model._res_lstm_block import ResLSTMBlock
from lmp.model._res_lstm_model import ResLSTMModel
from lmp.model._rnn import RNNModel
from lmp.model._self_attention_gru_model import SelfAttentionGRUModel
from lmp.model._self_attention_lstm_model import SelfAttentionLSTMModel
from lmp.model._self_attention_res_gru_model import SelfAttentionResGRUModel
from lmp.model._self_attention_res_lstm_model import SelfAttentionResLSTMModel

ALL_MODELS: Final[List[BaseModel]] = [
    RNNModel,
]
MODEL_OPTS: Final[Dict[str, BaseModel]] = {m.model_name: m for m in ALL_MODELS}
