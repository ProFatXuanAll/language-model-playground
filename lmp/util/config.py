r"""Helper function for setting or loading config.

Usage:
    config = lmp.util.load_optimizer(args)
"""
import torch
import os

from typing import Union

import lmp.config
import lmp.model

def load_config(args, file_path: str):
    if args.checkpoint > 0:
        config = lmp.config.BaseConfig.load_from_file(file_path)
    else:
        config = lmp.config.BaseConfig(
            batch_size=args.batch_size,
            checkpoint_step=args.checkpoint_step,
            dropout=args.dropout,
            embedding_dim=args.embedding_dim,
            epoch=args.epoch,
            hidden_dim=args.hidden_dim,
            is_uncased=args.is_uncased,
            learning_rate=args.learning_rate,
            max_norm=args.max_norm,
            min_count=args.min_count,
            model_class=args.model_class,
            num_rnn_layers=args.num_rnn_layers,
            num_linear_layers=args.num_linear_layers,
            seed=args.seed,
            tokenizer_class=args.tokenizer_class
        )


    return config

