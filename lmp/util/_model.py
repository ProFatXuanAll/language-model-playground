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

    Supported options:
        --model_class rnn
        --model_class gru
        --model_class lstm
        --model_class res_rnn
        --model_class res_gru
        --model_class res_lstm
        --model_class att_rnn
        --model_class att_gru
        --model_class att_lstm
        --model_class att_res_rnn
        --model_class att_res_gru
        --model_class att_res_lstm

    Load model from pre-trained checkpoint when `checkpoint != -1`.

    Args:
        checkpoint:
            Pre-trained model's checkpoint. Must be bigger than or equal to
            `-1`.
        d_emb:
            Embedding matrix vector dimension. Must be bigger than or equal to
            `1`.
        d_hid:
            Model layers hidden dimension. Must be bigger than or equal to
            `1`.
        device:
            Model running device.
        dropout:
            Dropout probability on all layers output (except output layer).
            Must range from `0.0` to `1.0`.
        experiment:
            Name of the pre-trained experiment. Must not be empty when
            `checkpoint != -1`.
        num_linear_layers:
            Number of Linear layers to use. Must be bigger than or equal to
            `1`.
        num_rnn_layers:
            Number of RNN layers to use. Must be bigger than or equal to
            `1`.
        pad_token_id:
            Padding token's id. Embedding layer will initialize padding
            token's vector with zeros. Must be bigger than or equal to `0`, and
            must be smaller than `vocab_size`.
        vocab_size:
            Embedding matrix vocabulary dimension. Must be bigger than or equal
            to `1`.

    Raises:
        TypeError:
            When one of the arguments are not an instance of their type
            annotation respectively.
        ValueError:
            When one of the arguments do not follow their constraints. See
            docstring for arguments constraints.

    Returns:
        `lmp.model.BaseRNNModel` if `model_class == 'rnn'`;
        `lmp.model.GRUModel` if `model_class == 'gru'`;
        `lmp.model.LSTMModel` if `model_class == 'lstm'`;
        `lmp.model.BaseResRNNModel` if `model_class == 'res_rnn'`;
        `lmp.model.ResGRUModel` if `model_class == 'res_gru'`;
        `lmp.model.ResLSTMModel` if `model_class == 'res_lstm'`.
        `lmp.model.BaseSelfAttentionRNNModel` if `model_class == 'att_rnn'`;
        `lmp.model.SelfAttentionGRUModel` if `model_class == 'att_gru'`;
        `lmp.model.SelfAttentionLSTMModel` if `model_class == 'att_lstm'`;
        `lmp.model.BaseSelfAttentionResRNNModel` if `model_class == 'att_res_rnn'`;
        `lmp.model.SelfAttentionResGRUModel` if `model_class == 'att_res_gru'`;
        `lmp.model.SelfAttentionResLSTMModel` if `model_class == 'att_res_lstm'`.
    """
    # Type check.
    if not isinstance(checkpoint, int):
        raise TypeError('`checkpoint` must be an instance of `int`.')

    if not isinstance(experiment, str):
        raise TypeError('`experiment` must be an instance of `str`.')

    if not isinstance(device, torch.device):
        raise TypeError('`device` must be an instance of `torch.device`.')

    if not isinstance(model_class, str):
        raise TypeError('`model_class` must be an instance of `str`.')

    # Value Check.
    if checkpoint < -1:
        raise ValueError('`checkpoint` must be bigger than or equal to `-1`.')

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

    elif model_class == 'att_rnn':
        model = lmp.model.BaseSelfAttentionRNNModel(
            d_emb=d_emb,
            d_hid=d_hid,
            dropout=dropout,
            num_rnn_layers=num_rnn_layers,
            num_linear_layers=num_linear_layers,
            pad_token_id=pad_token_id,
            vocab_size=vocab_size
        )

    elif model_class == 'att_gru':
        model = lmp.model.SelfAttentionGRUModel(
            d_emb=d_emb,
            d_hid=d_hid,
            dropout=dropout,
            num_rnn_layers=num_rnn_layers,
            num_linear_layers=num_linear_layers,
            pad_token_id=pad_token_id,
            vocab_size=vocab_size
        )

    elif model_class == 'att_lstm':
        model = lmp.model.SelfAttentionLSTMModel(
            d_emb=d_emb,
            d_hid=d_hid,
            dropout=dropout,
            num_rnn_layers=num_rnn_layers,
            num_linear_layers=num_linear_layers,
            pad_token_id=pad_token_id,
            vocab_size=vocab_size
        )

    elif model_class == 'att_res_rnn':
        model = lmp.model.BaseSelfAttentionResRNNModel(
            d_emb=d_emb,
            d_hid=d_hid,
            dropout=dropout,
            num_rnn_layers=num_rnn_layers,
            num_linear_layers=num_linear_layers,
            pad_token_id=pad_token_id,
            vocab_size=vocab_size
        )

    elif model_class == 'att_res_gru':
        model = lmp.model.SelfAttentionResGRUModel(
            d_emb=d_emb,
            d_hid=d_hid,
            dropout=dropout,
            num_rnn_layers=num_rnn_layers,
            num_linear_layers=num_linear_layers,
            pad_token_id=pad_token_id,
            vocab_size=vocab_size
        )

    elif model_class == 'att_res_lstm':
        model = lmp.model.SelfAttentionResLSTMModel(
            d_emb=d_emb,
            d_hid=d_hid,
            dropout=dropout,
            num_rnn_layers=num_rnn_layers,
            num_linear_layers=num_linear_layers,
            pad_token_id=pad_token_id,
            vocab_size=vocab_size
        )
    elif model_class == 'transformer':
        model = lmp.model.TransformerModel(
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
                lambda option: f'\n\t--model_class {option}',
                [
                    'rnn',
                    'gru',
                    'lstm',
                    'res_rnn',
                    'res_gru',
                    'res_lstm',
                    'att_rnn',
                    'att_gru',
                    'att_lstm',
                    'att_res_rnn',
                    'att_res_gru',
                    'att_res_lstm',
                    'transformer',
                ]
            )))
        )

    if checkpoint != -1:
        if not experiment:
            raise ValueError('`experiment` must not be empty.')

        file_path = os.path.join(
            lmp.path.DATA_PATH,
            experiment,
            f'model-{checkpoint}.pt'
        )

        if not os.path.exists(file_path):
            raise FileNotFoundError(f'File {file_path} does not exist.')

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
            Pre-trained model's checkpoint. Must be bigger than or equal to
            `-1`.
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
        ValueError:
            When `checkpoint < -1` or `config.model_class` does not support.

    Returns:
        Same as `load_model`.
    """
    # Type check.
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
