from .base_model import BaseModel

import torch


class LSTMModel(BaseModel):
    r"""LSTM replaces BaseModel's rnn_layer.
    """

    def __init__(self, config, tokenizer):
        super(LSTMModel, self).__init__(config, tokenizer)
        # rewrite RNN layer
        self.rnn_layer = torch.nn.LSTM(input_size=config.embedding_dim,
                                       hidden_size=config.hidden_dim,
                                       num_layers=config.num_rnn_layers,
                                       dropout=config.dropout,
                                       batch_first=True)
