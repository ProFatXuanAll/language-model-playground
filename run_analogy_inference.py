
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
        '--test_words',
        help='Input 3 word in sequence word_a,word_b,word_c to calculate (word_b-word_a+word_c).Example: Taiwan,Taipei,Japan',
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

    # Get test data.
    test_data = [word for word in args.test_words.split(',')]

    # Test syntatic and semantic score.
    lmp.util.analogy_inference(
        test_data=test_data,
        device=config.device,
        model=model,
        tokenizer=tokenizer
    )
