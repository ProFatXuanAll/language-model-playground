r"""Train the text-generation model.
Usage:
    python example_train.py --experiment_no 1 --is_uncased True

--experiment_no is a required argument.
Run 'python example_train.py --help' for help
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
        '--experiment',
        help='Current experiment name.',
        required=True,
        type=str
    )

    # Optional arguments.
    parser.add_argument(
        '--batch_size',
        default=32,
        help='Training batch size.',
        type=int
    )
    parser.add_argument(
        '--checkpoint',
        default=-1,
        help='Start from specific checkpoint.',
        type=int
    )
    parser.add_argument(
        '--checkpoint_step',
        default=500,
        help='Checkpoint save interval.',
        type=int
    )
    parser.add_argument(
        '--d_emb',
        default=100,
        help='Embedding dimension.',
        type=int
    )
    parser.add_argument(
        '--d_hid',
        default=300,
        help='Hidden dimension.',
        type=int
    )
    parser.add_argument(
        '--dataset',
        default='news_collection',
        help='Name of the dataset to perform experiment.',
        type=str
    )
    parser.add_argument(
        '--dropout',
        default=0.0,
        help='Dropout rate.',
        type=float
    )
    parser.add_argument(
        '--epoch',
        default=10,
        help='Number of training epochs.',
        type=int
    )
    parser.add_argument(
        '--is_uncased',
        action='store_true',
        help='Whether to convert text from upper cases to lower cases.'
    )
    parser.add_argument(
        '--learning_rate',
        default=1e-4,
        help='Gradient decent learning rate.',
        type=float
    )
    parser.add_argument(
        '--max_norm',
        default=1.0,
        help='Gradient bound to avoid gradient explosion.',
        type=float
    )
    parser.add_argument(
        '--max_seq_len',
        default=64,
        help='Text sample max length.',
        type=int
    )
    parser.add_argument(
        '--min_count',
        default=1,
        help='Filter out tokens occur less than `min_count`.',
        type=int
    )
    parser.add_argument(
        '--model_class',
        default='lstm',
        help="Language model's class.",
        type=str
    )
    parser.add_argument(
        '--num_linear_layers',
        default=2,
        help='Number of Linear layers.',
        type=int
    )
    parser.add_argument(
        '--num_rnn_layers',
        default=1,
        help='Number of rnn layers.',
        type=int
    )
    parser.add_argument(
        '--optimizer_class',
        default='adam',
        help="Optimizer's class.",
        type=str
    )
    parser.add_argument(
        '--seed',
        default=7,
        help='Control random seed.',
        type=int
    )
    parser.add_argument(
        '--tokenizer_class',
        default='whitespace_list',
        help="Tokenizer's class.",
        type=str
    )

    args = parser.parse_args()

    # Hyperparameters setup.
    config = lmp.util.load_config(args)
    config.save()

    # Get model running device.
    device = config.device

    # Initialize random seed.
    lmp.util.set_seed_by_config(
        config=config
    )

    # Load data.
    dataset = lmp.util.load_dataset_by_config(
        config=config
    )

    # Load tokenizer.
    tokenizer = lmp.util.load_tokenizer_by_config(
        checkpoint=args.checkpoint,
        config=config
    )

    # Train tokenizer from scratch if necessary.
    if args.checkpoint == -1:
        lmp.util.train_tokenizer_by_config(
            config=config,
            dataset=dataset,
            tokenizer=tokenizer
        )
        tokenizer.save(experiment=config.experiment)

    # Load model.
    model = lmp.util.load_model_by_config(
        checkpoint=args.checkpoint,
        config=config,
        tokenizer=tokenizer
    )

    # Load optimizer
    optimizer = lmp.util.load_optimizer_by_config(
        checkpoint=args.checkpoint,
        config=config,
        model=model
    )

    # Train model.
    lmp.util.train_model_by_config(
        checkpoint=args.checkpoint,
        config=config,
        dataset=dataset,
        model=model,
        optimizer=optimizer,
        tokenizer=tokenizer
    )
