r"""Calculating perplexities on dataset.

Usage:
    python run_perplexity_evaluation.py ...

Run 'python run_perplexity_evaluation.py --help' for help.
"""
# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse

# self-made modules

import lmp

if __name__ == '__main__':
    # Parse argument from standard input.
    parser = argparse.ArgumentParser()

    # Required arguments.
    parser.add_argument(
        '--checkpoint',
        help='Load specific checkpoint.',
        required=True,
        type=int
    )
    parser.add_argument(
        '--dataset',
        help='Current experiment name.',
        required=True,
        type=str,
    )
    parser.add_argument(
        '--experiment',
        help='Current experiment name.',
        required=True,
        type=str,
    )

    args = parser.parse_args()

    # Load pre-trained hyperparameters.
    config = lmp.config.BaseConfig.load(experiment=args.experiment)

    # Overwrite evaluation dataset.
    config.dataset = args.dataset

    # Load dataset.
    dataset = lmp.util.load_dataset_by_config(config=config)

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

    # Calculating dataset perplexity.
    perplexities = lmp.util.batch_perplexity_eval(
        dataset=dataset,
        device=config.device,
        model=model,
        tokenizer=tokenizer
    )

    # Print perplexity of each sequence.
    for i, perplexity in enumerate(perplexities):
        print(f'{perplexity:.6f}, {dataset[i]}')
