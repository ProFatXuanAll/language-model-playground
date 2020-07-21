r"""Helper function for loading optimizer.

Usage:
    optimizer = lmp.util.load_optimizer(config = config)
"""
import torch
import os

from typing import Union

import lmp.config
import lmp.model


def load_optimizer(
    checkpoint: int,
    config: lmp.config.BaseConfig,
    model: Union[lmp.model.GRUModel, lmp.model.LSTMModel],
    save_path: str

):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    if checkpoint > 0:
        state_path = f'{save_path}/checkpoint{checkpoint}.pt'
        checkpoint_state = torch.load(state_path)

        optimizer.load_state_dict(checkpoint_state['optimizer'])

    return optimizer
