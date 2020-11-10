r"""Language model module.

All model must import from this file.

Usage:
    import lmp.model

    model = lmp.model.BaseResRNNModel(...)
    model = lmp.model.BaseRNNModel(...)
    model = lmp.model.BaseSelfAttentionResRNNModel(...)
    model = lmp.model.BaseSelfAttentionRNNModel(...)
    model = lmp.model.GRUModel(...)
    model = lmp.model.LSTMModel(...)
    model = lmp.model.ResGRUModel(...)
    model = lmp.model.ResLSTMModel(...)
    model = lmp.model.SelfAttentionGRUModel(...)
    model = lmp.model.SelfAttentionLSTMModel(...)
    model = lmp.model.SelfAttentionResGRUModel(...)
    model = lmp.model.SelfAttentionResLSTMModel(...)
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# self-made modules

from lmp.model._attention_mechanism import attention_mechanism
from lmp.model._base_res_rnn_block import BaseResRNNBlock
from lmp.model._base_res_rnn_model import BaseResRNNModel
from lmp.model._base_rnn_model import BaseRNNModel
from lmp.model._base_self_attention_res_rnn_model import BaseSelfAttentionResRNNModel
from lmp.model._base_self_attention_rnn_model import BaseSelfAttentionRNNModel
from lmp.model._self_attention_gru_model import SelfAttentionGRUModel
from lmp.model._self_attention_lstm_model import SelfAttentionLSTMModel
from lmp.model._self_attention_res_gru_model import SelfAttentionResGRUModel
from lmp.model._self_attention_res_lstm_model import SelfAttentionResLSTMModel
from lmp.model._gru_model import GRUModel
from lmp.model._lstm_model import LSTMModel
from lmp.model._res_gru_block import ResGRUBlock
from lmp.model._res_gru_model import ResGRUModel
from lmp.model._res_lstm_block import ResLSTMBlock
from lmp.model._res_lstm_model import ResLSTMModel
from lmp.model._transformer_model import TransformerModel
