r"""Helper function for loading optimizer.

Usage:
    import lmp

    optimizer = lmp.util.load_optimizer(...)
    optimizer = lmp.util.load_optimizer_by_config(...)
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

from typing import Iterator
from typing import Union

# 3rd-party modules

import torch

# self-made modules

import lmp.config
import lmp.model
import lmp.path


def load_optimizer(
        checkpoint: int,
        experiment: str,
        learning_rate: float,
        optimizer_class: str,
        parameters: Iterator[torch.nn.Parameter]
) -> Union[
    torch.optim.SGD,
    torch.optim.Adam,
]:
    r"""Helper function for constructing optimizer.

    Load optimizer from pre-trained checkpoint when `checkpoint != -1`.

    Args:
        checkpoint:
            Pre-trained optimizer's checkpoint.
        experiment:
            Name of the pre-trained experiment.
        learning_rate:
            Gradient descend learning rate.
        optimizer_class:
            Optimizer's class.
        parameters:
            Model parameters to be optimized.

    Raises:
        ValueError:
            If `optimizer` does not supported.

    Returns:
        `torch.optim.SGD` if `optimizer_class == 'sgd'`;
        `torch.optim.Adam` if `optimizer_class == 'adam'`.
    """
    if optimizer_class == 'sgd':
        optimizer = torch.optim.SGD(
            params=parameters,
            lr=learning_rate
        )
    elif optimizer_class == 'adam':
        optimizer = torch.optim.Adam(
            params=parameters,
            lr=learning_rate
        )
    else:
        raise ValueError(
            f'`{optimizer_class}` does not support\nSupported options:' +
            ''.join(list(map(
                lambda option: f'\n\t--optimizer {option}',
                [
                    'sgd',
                    'adam',
                ]
            )))
        )

    if checkpoint != -1:
        file_path = f'{lmp.path.DATA_PATH}/{experiment}/optimizer-{checkpoint}.pt'
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'file {file_path} does not exist.')
        optimizer.load_state_dict(torch.load(file_path))

    return optimizer


def load_optimizer_by_config(
        checkpoint: int,
        config: lmp.config.BaseConfig,
        model: lmp.model.BaseRNNModel
) -> Union[
    torch.optim.SGD,
    torch.optim.Adam,
]:
    r"""Helper function for constructing optimizer.

    Load optimizer from pre-trained checkpoint when `checkpoint != -1`.

    Args:
        checkpoint:
            Pre-trained optimizer's checkpoint.
        config:
            Configuration object with attributes `experiment`, `learning_rate`
            and `optimizer_class`.
        model:
            Source of model parameters.

    Returns:
        Same as `load_optimizer`.
    """
    return load_optimizer(
        checkpoint=checkpoint,
        experiment=config.experiment,
        learning_rate=config.learning_rate,
        optimizer_class=config.optimizer_class,
        parameters=model.parameters()
    )
