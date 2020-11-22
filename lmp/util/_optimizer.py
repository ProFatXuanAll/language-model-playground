r"""Helper function for loading optimizer.

Usage:
    import lmp.util

    optimizer = lmp.util.load_optimizer(...)
    optimizer = lmp.util.load_optimizer_by_config(...)
"""



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import os

from typing import Iterable
from typing import Union



import torch



import lmp.config
import lmp.model
import lmp.path


def load_optimizer(
        checkpoint: int,
        experiment: str,
        learning_rate: float,
        optimizer_class: str,
        parameters: Iterable[torch.nn.Parameter]
) -> Union[torch.optim.SGD, torch.optim.Adam]:
    r"""Helper function for constructing optimizer.

    Supported options:
        --optimizer_class sgd
        --optimizer_class adam

    Load optimizer from pre-trained checkpoint when `checkpoint != -1`.

    Args:
        checkpoint:
            Pre-trained model's checkpoint. Must be bigger than or equal to
            `-1`.
        experiment:
            Name of the pre-trained experiment. Must not be empty when
            `checkpoint != -1`.
        learning_rate:
            Gradient descend learning rate. Must be bigger than `0.0`.
        optimizer_class:
            Optimizer's class.
        parameters:
            Model parameters to be optimized. Must not be empty.

    Raises:
        TypeError:
            When one of the arguments are not an instance of their type
            annotation respectively.
        ValueError:
            When one of the arguments do not follow their constraints. See
            docstring for arguments constraints.

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
            '`Iterable[torch.nn.Parameter]`.'
        )

    parameters = list(parameters)

    if not all(map(lambda x: isinstance(x, torch.nn.Parameter), parameters)):
        raise TypeError(
            '`parameters` must be an instance of '
            '`Iterable[torch.nn.Parameter]`.',
        )

    # Value Check.
    if checkpoint < -1:
        raise ValueError('`checkpoint` must be bigger than or equal to `-1`.')

    if learning_rate < 0.0 or math.isnan(learning_rate):
        raise ValueError('`learning_rate` must be bigger than `0.0`.')

    if not parameters:
        raise ValueError('`parameters` must not be empty.')

    if optimizer_class == 'sgd':
        optimizer = torch.optim.SGD(params=parameters, lr=learning_rate)

    elif optimizer_class == 'adam':
        optimizer = torch.optim.Adam(params=parameters, lr=learning_rate)

    else:
        raise ValueError(
            f'optimizer `{optimizer_class}` does not support\n' +
            'Supported options:' +
            ''.join(list(map(
                lambda option: f'\n\t--optimizer_class {option}',
                [
                    'sgd',
                    'adam',
                ]
            )))
        )

    if checkpoint != -1:
        if not experiment:
            raise ValueError('`experiment` must not be empty.')

        file_path = os.path.join(
            lmp.path.DATA_PATH,
            experiment,
            f'optimizer-{checkpoint}.pt'
        )

        if not os.path.exists(file_path):
            raise FileNotFoundError(f'File {file_path} does not exist.')

        optimizer.load_state_dict(torch.load(file_path))

    return optimizer


def load_optimizer_by_config(
        checkpoint: int,
        config: lmp.config.BaseConfig,
        model: Union[lmp.model.BaseRNNModel, lmp.model.BaseResRNNModel]
) -> Union[torch.optim.SGD, torch.optim.Adam]:
    r"""Helper function for constructing optimizer.

    Load optimizer from pre-trained checkpoint when `checkpoint != -1`.

    Args:
        checkpoint:
            Pre-trained model's checkpoint. Must be bigger than or equal to
            `-1`.
        config:
            Configuration object with attributes `experiment`, `learning_rate`
            and `optimizer_class`.
        model:
            Source of model parameters.

    Raises:
        TypeError:
            When one of the arguments are not an instance of their type
            annotation respectively.
        ValueError:
            When `checkpoint < -1`.

    Returns:
        Same as `load_optimizer`.
    """
    # Type check.
    if not isinstance(config, lmp.config.BaseConfig):
        raise TypeError(
            '`config` must be an instance of `lmp.config.BaseConfig`.'
        )

    if not isinstance(model, (
            lmp.model.BaseRNNModel,
            lmp.model.BaseResRNNModel,
            lmp.model.TransformerModel
    )):
        raise TypeError(
            '`model` must be an instance of '
            '`Union[lmp.model.BaseRNNModel, lmp.model.BaseResRNNModel]`.'
        )

    return load_optimizer(
        checkpoint=checkpoint,
        experiment=config.experiment,
        learning_rate=config.learning_rate,
        optimizer_class=config.optimizer_class,
        parameters=model.parameters()
    )
