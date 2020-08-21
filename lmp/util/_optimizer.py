r"""Helper function for loading optimizer.

Usage:
    import lmp.util

    optimizer = lmp.util.load_optimizer(...)
    optimizer = lmp.util.load_optimizer_by_config(...)
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

from typing import Iterable
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
        TypeError:
            When `checkpoint` is not an instance of `int`, `experiment` is not
            an instance of `str`, `learning_rate` is not an instance of `float`
            , `optimizer_class` is not an instance of `str` or `parameters` is
            not an instance of `Iterator[torch.nn.Parameter]`.
        ValueError:
            If `optimizer` does not supported.

    Returns:
        `torch.optim.SGD` if `optimizer_class == 'sgd'`;
        `torch.optim.Adam` if `optimizer_class == 'adam'`.
    """
    # Type check.
    if not isinstance(checkpoint, int):
        raise TypeError('`checkpoint` must be an instance of `int`.')
    
    if not isinstance(experiment, str):
        raise TypeError('`experiment` must be an instance of `str`.')

    if not isinstance(learning_rate, float):
        raise TypeError('`learning_rate` must be an instance of `float`.')

    if not isinstance(optimizer_class, str):
        raise TypeError('`optimizer_class` must be an instance of `str`.')

    if not isinstance(parameters, Iterable):
        raise TypeError(
            '`parameters` must be an instance of '
            '`Iterator[torch.nn.Parameter]`.'
        )
    parameters = list(parameters)
    if not all(map(lambda x: isinstance(x, torch.nn.Parameter), parameters)):
        raise TypeError(
            '`parameters` must be an instance of '
            '`Iterator[torch.nn.Parameter]`.',
        )


    if optimizer_class == 'sgd':
        optimizer = torch.optim.SGD(params=parameters, lr=learning_rate)

    elif optimizer_class == 'adam':
        optimizer = torch.optim.Adam(params=parameters, lr=learning_rate)

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
        model: Union[
            lmp.model.BaseRNNModel,
            lmp.model.BaseResRNNModel
        ]
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

    Raises:
        TypeError:
            When `config` is not an instance of `lmp.config.BaseConfig` or
            `model` is not an instance of `lmp.model.BaseRNNModel` and
            `BaseResRNNModel`.

    Returns:
        Same as `load_optimizer`.
    """
    # Type check.
    if not isinstance(config, lmp.config.BaseConfig):
        raise TypeError(
            '`config` must be an instance of `lmp.config.BaseConfig`.'
        )

    if not isinstance(model, lmp.model.BaseRNNModel) and not isinstance(
        model,
        lmp.model.BaseResRNNModel
    ):
        raise TypeError(
            '`model` must be an instance of '
            '`Union['
                'lmp.model.BaseRNNModel,'
                'lmp.model.BaseResRNNModel'
            ']`.'
        )

    return load_optimizer(
        checkpoint=checkpoint,
        experiment=config.experiment,
        learning_rate=config.learning_rate,
        optimizer_class=config.optimizer_class,
        parameters=model.parameters()
    )
