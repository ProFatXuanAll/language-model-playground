from .base_model import BaseModel

import torch

class GRUModel(BaseModel):
    r"""GRU replaces BaseModel's rnn_layer.
    """

    def __init__(self, config, tokenizer):
        super(GRUModel, self).__init__(config, tokenizer)

        # rewrite RNN layer
        self.rnn_layer = torch.nn.GRU(input_size=config.embedding_dim,
                                      hidden_size=config.hidden_dim,
                                      num_layers=config.num_rnn_layers,
                                      dropout=config.dropout,
                                      batch_first=True)
