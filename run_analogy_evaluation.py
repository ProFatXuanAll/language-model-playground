r"""Calculating accuracy on word analogy dataset.

Usage:
    python run_analogy_evaluation.py ...

Run 'python run_analogy_evaluation.py --help' for help.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse



import torch



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
        help='Name of the dataset to perform analogy.',
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

    # Calculating word analogy dataset accuracy per category.
    acc_per_cat = lmp.util.analogy_eval(
        dataset=dataset,
        device=config.device,
        model=model,
        tokenizer=tokenizer
    )

    for category in acc_per_cat:
        print(f'category: {category}, accuracy: {acc_per_cat[category]}')
