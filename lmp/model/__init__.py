r"""Language model module.

All model must be import from this file.

Usage:
    import lmp.model

    model = lmp.model.BaseRNNModel(...)
    model = lmp.model.GRUModel(...)
    model = lmp.model.LSTMModel(...)
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# self-made modules

from lmp.model._base_res_rnn_block import BaseResRNNBlock
from lmp.model._base_residual_rnn_model import BaseResidualRNNModel
from lmp.model._base_rnn_model import BaseRNNModel
from lmp.model._gru_model import GRUModel
from lmp.model._lstm_model import LSTMModel
from lmp.model._res_gru_block import ResGRUBlock
from lmp.model._res_lstm_block import ResLSTMBlock
from lmp.model._residual_gru_model import ResidualGRUModel
from lmp.model._residual_lstm_model import ResidualLSTMModel

