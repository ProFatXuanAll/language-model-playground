r"""Train language model.

Usage:
    python run_train.py ...

Run 'python run_train.py --help' for help.
"""



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import time



import torch



import lmp

if __name__ == '__main__':
    # Record total execution time.
    start_time = time.time()

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
        default='news_collection_title',
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

    # Create collate_fn for sampling.
    collate_fn = lmp.dataset.LanguageModelDataset.create_collate_fn(
        tokenizer=tokenizer,
        max_seq_len=config.max_seq_len
    )

    # `torch` utility for sampling.
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    # Train model.
    lmp.util.train_model(
        checkpoint=args.checkpoint,
        checkpoint_step=config.checkpoint_step,
        data_loader=data_loader,
        device=config.device,
        epoch=config.epoch,
        experiment=config.experiment,
        max_norm=config.max_norm,
        model=model,
        optimizer=optimizer,
        vocab_size=tokenizer.vocab_size
    )

    total_exec_time = time.time() - start_time
    print('Total execution time: {} hrs {} mins {} secs'.format(
        int(total_exec_time // 3600),
        int(total_exec_time // 60),
        int(total_exec_time % 60)
    ))
