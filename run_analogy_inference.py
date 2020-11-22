r"""Giving `word_a`, `word_b`, `word_c` to generate `word_d`.

Usage:
    python run_analogy_inference.py ...

Run 'python run_analogy_inference.py --help' for help.
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
        '--experiment',
        help='Current experiment name.',
        required=True,
        type=str,
    )
    parser.add_argument(
        '--word_a',
        help='`word_a` to perform word analogy (`word_b - word_a + word_c`).',
        required=True,
        type=str,
    )
    parser.add_argument(
        '--word_b',
        help='`word_b` to perform word analogy (`word_b - word_a + word_c`).',
        required=True,
        type=str,
    )
    parser.add_argument(
        '--word_c',
        help='`word_c` to perform word analogy (`word_b - word_a + word_c`).',
        required=True,
        type=str,
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

    # Generate analog word
    word_d = lmp.util.analogy_inference(
        device=config.device,
        model=model,
        tokenizer=tokenizer,
        word_a=args.word_a,
        word_b=args.word_b,
        word_c=args.word_c
    )

    print(f'{args.word_a} : {args.word_b} = {args.word_c} : {word_d}')
