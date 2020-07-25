r"""Helper function for configuration construction.

Usage:
    import lmp

    config = lmp.util.load_optimizer(args)
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse

# self-made modules

import lmp.config
import lmp.model


def load_config(args: argparse.Namespace) -> lmp.config.BaseConfig:
    r"""Load configuration from standard input.

    Args:
        args:
            Standard input argument parser object with attributes `batch_size`,
            `checkpoint_step`, `d_emb`, `d_hid`, `dataset`, `dropout`, `epoch`,
            `experiment`, `is_uncased`, `learning_rate`, `max_norm`,
            `max_seq_len`, `min_count`, `model_class`, `num_linear_layers`,
            `num_rnn_layers`, `optimizer_class`, `seed` and `tokenizer_class`.

    Returns:
        Configuration object which can be used with most of utilities.
    """
    if args.checkpoint != -1:
        config = lmp.config.BaseConfig.load(experiment=args.experiment)
    else:
        config = lmp.config.BaseConfig(
            batch_size=args.batch_size,
            checkpoint_step=args.checkpoint_step,
            d_emb=args.d_emb,
            d_hid=args.d_hid,
            dataset=args.dataset,
            dropout=args.dropout,
            epoch=args.epoch,
            experiment=args.experiment,
            is_uncased=args.is_uncased,
            learning_rate=args.learning_rate,
            max_norm=args.max_norm,
            max_seq_len=args.max_seq_len,
            min_count=args.min_count,
            model_class=args.model_class,
            num_linear_layers=args.num_linear_layers,
            num_rnn_layers=args.num_rnn_layers,
            optimizer_class=args.optimizer_class,
            seed=args.seed,
            tokenizer_class=args.tokenizer_class
        )
    if args.epoch != config.epoch:
        config.epoch = args.epoch

    return config
