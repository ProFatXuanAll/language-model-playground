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


if __name__ == "__main__":
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
    dataset=lmp.analogy_dataset.AnalogyDataset()
    dataloader= torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        # shuffle=True,
    )
    # test syntatic and semantic score
    lmp.util.syntatic_and_semantic_test(
        device=config.device,
        model=model,
        data_loader=dataloader,
        tokenizer=tokenizer
    )

