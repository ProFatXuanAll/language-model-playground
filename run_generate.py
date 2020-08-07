r"""Giving a text to generate rest sequence.

Usage:
    python run_generate.py ...

Run 'python run_generate.py --help' for help.
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse

# 3rd-party modules

import torch

# self-made modules

import lmp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required arguments.
    parser.add_argument(
        '--begin_of_sequence',
        help='Begining of sequence which model will auto-complete.',
        required=True,
        type=str
    )
    parser.add_argument(
        '--checkpoint',
        help='Load specific checkpoint.',
        required=True,
        type=int
    )
    parser.add_argument(
        '--experiment',
        help='Current experiment name.',
        required=True,
        type=str,
    )

    # Optional arguments.
    parser.add_argument(
        '--beam_width',
        default=4,
        help='using for generating `beam_width` sentences',
        type=int
    )
    parser.add_argument(
        '--max_seq_len',
        default=64,
        help='Text sample max length.',
        type=int
    )

    args = parser.parse_args()

    # Load pre-trained hyperparameters.
    config = lmp.config.BaseConfig.load(experiment=args.experiment)

    # Load pre-trained tokenizer.
    tokenizer = lmp.util.load_tokenizer_by_config(
        checkpoint=args.checkpoint,
        config=config
    )

    # Load pre-trained model.
    model = lmp.util.load_model_by_config(
        checkpoint=args.checkpoint,
        config=config,
        tokenizer=tokenizer
    )

    # Sequences generation.
    generated_sequences = lmp.util.generate_sequence_by_config(
        beam_width=args.beam_width,
        begin_of_sequence=args.begin_of_sequence,
        config=config,
        max_seq_len=args.max_seq_len,
        model=model,
        tokenizer=tokenizer
    )

    # Output generated sequences.
    for sequence in generated_sequences:
        print(sequence)
