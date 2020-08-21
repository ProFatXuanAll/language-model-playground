r"""Helper function for loading model.

Usage:
    import lmp.util

    model = lmp.util.load_model(...)
    model = lmp.util.load_model_by_config(...)
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

from typing import Union

# 3rd-party modules

import torch

# self-made modules

import lmp.config
import lmp.model
import lmp.tokenizer


def load_model(
        checkpoint: int,
        d_emb: int,
        d_hid: int,
        device: torch.device,
        dropout: float,
        experiment: str,
        model_class: str,
        num_linear_layers: int,
        num_rnn_layers: int,
        pad_token_id: int,
        vocab_size: int
) -> Union[lmp.model.BaseRNNModel, lmp.model.BaseResRNNModel]:
    r"""Helper function for constructing language model.

    Load optimizer from pre-trained checkpoint when `checkpoint != -1`.

    Args:
        checkpoint:
            Pre-trained model's checkpoint.
        d_emb:
            Embedding matrix vector dimension.
        d_hid:
            Model layers hidden dimension.
        device:
            Model running device.
        dropout:
            Dropout probability on all layers output (except output layer).
        experiment:
            Name of the pre-trained experiment.
        num_rnn_layers:
            Number of GRU layers to use.
        num_linear_layers:
            Number of Linear layers to use.
        pad_token_id:
            Padding token's id. Embedding layers will initialize padding
            token's vector with zeros.
        vocab_size:
            Embedding matrix vocabulary dimension.

    Raises:
        TypeError:
            When `checkpoint` is not an instance of `int`, `d_emb` is not an
            instance of `int`, `d_hid` is not an instance of `int`, `device`
            is not an instance of `torch.device`, `dropout` is not an instance
            of `float`, `experiment` is not an instance of `str`, `model_class`
            is not an instance of `str`, `num_linear_layers` is not an instance
            of `int`, `num_rnn_layers` is not an instance of `int`,
            `pad_token_id` is not an instance of `int` or `vocab_size` is not
            an instance of `int`.
        ValueError:
            If `model` does not supported, `d_emb` is samller than `1`, `d_hid`
            is samller than `1`, `dropout` is not in range[0,1], `experiment`
            is empty, `model_class` is empty, `num_linear_layers` is smaller
            than `1`, `num_rnn_layers` is smaller than `1` or `pad_token_id`
            is smaller than `0`.  


    Returns:
        `lmp.model.BaseRNNModel` if `model_class == 'rnn'`;
        `lmp.model.GRUModel` if `model_class == 'gru'`;
        `lmp.model.LSTMModel` if `model_class == 'lstm'`;
        `lmp.model.BaseResRNNModel` if `model_class == 'res_rnn'`;
        `lmp.model.ResGRUModel` if `model_class == 'res_gru'`;
        `lmp.model.ResLSTMModel` if `model_class == 'res_lstm'`.
    """
    # Type check.
    if not isinstance(checkpoint, int):
        raise TypeError('`checkpoint` must be an instance of `int`.')
    
    if not isinstance(d_emb, int):
        raise TypeError('`d_emb` must be an instance of `int`.')
    
    if not isinstance(d_hid, int):
        raise TypeError('`d_hid` must be an instance of `int`.')
    
    if not isinstance(device, torch.device):
        raise TypeError('`device` must be an instance of `torch.device`.')

    if not isinstance(dropout, float):
        raise TypeError('`dropout` must be an instance of `float`.')

    if not isinstance(experiment, str):
        raise TypeError('`experiment` must be an instance of `str`.')

    if not isinstance(model_class, str):
        raise TypeError('`model_class` must be an instance of `str`.')

    if not isinstance(num_linear_layers, int):
        raise TypeError('`num_linear_layers` must be an instance of `int`.')

    if not isinstance(num_rnn_layers, int):
        raise TypeError('`num_rnn_layers` must be an instance of `int`.')

    if not isinstance(pad_token_id, int):
        raise TypeError('`pad_token_id` must be an instance of `int`.')

    if not isinstance(vocab_size, int):
        raise TypeError('`vocab_size` must be an instance of `int`.')

    # Value Check.
    if d_emb < 1:
        raise ValueError('`d_emb` must be bigger than or equal to `1`.')

    if d_hid < 1:
        raise ValueError('`d_hid` must be bigger than or equal to `1`.')

    if not 0 <= dropout <= 1:
        raise ValueError('`dropout` must range from `0.0` to `1.0`.')

    if not experiment:
        raise ValueError('`experiment` must not be empty.')

    if not model_class:
        raise ValueError('`model_class` must not be empty.')

    if num_linear_layers < 1:
        raise ValueError(
            '`num_linear_layers` must be bigger than or equal to `1`.'
        )

    if num_rnn_layers < 1:
        raise ValueError(
            '`num_rnn_layers` must be bigger than or equal to `1`.'
        )
    
    if pad_token_id < 0:
        raise ValueError(
            '`pad_token_id` must be bigger than or equal to `0`.'
        )


    if model_class == 'rnn':
        model = lmp.model.BaseRNNModel(
            d_emb=d_emb,
            d_hid=d_hid,
            dropout=dropout,
            num_rnn_layers=num_rnn_layers,
            num_linear_layers=num_linear_layers,
            pad_token_id=pad_token_id,
            vocab_size=vocab_size
        )

    elif model_class == 'gru':
        model = lmp.model.GRUModel(
            d_emb=d_emb,
            d_hid=d_hid,
            dropout=dropout,
            num_rnn_layers=num_rnn_layers,
            num_linear_layers=num_linear_layers,
            pad_token_id=pad_token_id,
            vocab_size=vocab_size
        )

    elif model_class == 'lstm':
        model = lmp.model.LSTMModel(
            d_emb=d_emb,
            d_hid=d_hid,
            dropout=dropout,
            num_rnn_layers=num_rnn_layers,
            num_linear_layers=num_linear_layers,
            pad_token_id=pad_token_id,
            vocab_size=vocab_size
        )

    elif model_class == 'res_rnn':
        model = lmp.model.BaseResRNNModel(
            d_emb=d_emb,
            d_hid=d_hid,
            dropout=dropout,
            num_rnn_layers=num_rnn_layers,
            num_linear_layers=num_linear_layers,
            pad_token_id=pad_token_id,
            vocab_size=vocab_size
        )

    elif model_class == 'res_gru':
        model = lmp.model.ResGRUModel(
            d_emb=d_emb,
            d_hid=d_hid,
            dropout=dropout,
            num_rnn_layers=num_rnn_layers,
            num_linear_layers=num_linear_layers,
            pad_token_id=pad_token_id,
            vocab_size=vocab_size
        )

    elif model_class == 'res_lstm':
        model = lmp.model.ResLSTMModel(
            d_emb=d_emb,
            d_hid=d_hid,
            dropout=dropout,
            num_rnn_layers=num_rnn_layers,
            num_linear_layers=num_linear_layers,
            pad_token_id=pad_token_id,
            vocab_size=vocab_size
        )

    else:
        raise ValueError(
            f'model `{model_class}` does not support.\nSupported options:' +
            ''.join(list(map(
                lambda option: f'\n\t--model {option}',
                [
                    'rnn',
                    'gru',
                    'lstm',
                    'res_rnn',
                    'res_gru',
                    'res_lstm',
                ]
            )))
        )

    if checkpoint != -1:
        file_path = f'{lmp.path.DATA_PATH}/{experiment}/model-{checkpoint}.pt'
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'file {file_path} does not exist.')
        model.load_state_dict(torch.load(file_path))

    return model.to(device)


def load_model_by_config(
        checkpoint: int,
        config: lmp.config.BaseConfig,
        tokenizer: lmp.tokenizer.BaseTokenizer
) -> Union[lmp.model.BaseRNNModel, lmp.model.BaseResRNNModel]:
    r"""Helper function for constructing language model.

    Load model from pre-trained checkpoint when `checkpoint != -1`.

    Args:
        checkpoint:
            Pre-trained model's checkpoint.
        config:
            Configuration object with attributes `d_emb`, `d_hid`, `dropout`,
            `device`, `experiment`, `model_class`, `num_linear_layer` and
            `num_rnn_layer`.
        tokenizer:
            Tokenizer object with attributes `pad_token_id` and `vocab_size`.

    Raises:
        TypeError:
            When `config` is not an instance of `lmp.config.BaseConfig` or
            `tokenizer` is not an instance of `lmp.tokenizer.BaseTokenizer`.

    Returns:
        Same as `load_model`.
    """
    #Type check.
    if not isinstance(config, lmp.config.BaseConfig):
        raise TypeError(
            '`config` must be an instance of `lmp.config.BaseConfig`.'
        )
    
    if not isinstance(tokenizer, lmp.tokenizer.BaseTokenizer):
        raise TypeError(
            '`tokenizer` must be an instance of `lmp.tokenizer.BaseTokenizer`.'
        )
    
    return load_model(
        checkpoint=checkpoint,
        d_emb=config.d_emb,
        d_hid=config.d_hid,
        device=config.device,
        dropout=config.dropout,
        experiment=config.experiment,
        model_class=config.model_class,
        num_linear_layers=config.num_linear_layers,
        num_rnn_layers=config.num_rnn_layers,
        pad_token_id=tokenizer.convert_token_to_id(tokenizer.pad_token),
        vocab_size=tokenizer.vocab_size
    )
