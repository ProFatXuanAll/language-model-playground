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

from lmp.model._base_rnn_model import BaseRNNModel
from lmp.model._gru_model import GRUModel
from lmp.model._lstm_model import LSTMModel
