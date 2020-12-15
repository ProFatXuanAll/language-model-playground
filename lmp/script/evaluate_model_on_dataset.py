r"""Evaluate language model on dataset.

Tool for evaluating language model on dataset.
Pre-trained model are used to calculate perplexity on dataset.
This script serve as model validation, and it is usually run after training
model.
All evaluation results will be shown on both CLI and tensorboard.
Use ``pipenv run tensorboard`` to launch tensorboard and use browser to open
URL http://localhost:6006/ to see evaluation results.

See Also
========
lmp.model
    All available models.
lmp.script.train_model
    Train language model.

Examples
========
The following example using ``valid`` version of :py:class:`lmp.dset.WikiText2`
dataset to evaluate pre-trained language model experiment ``my_exp``.
It evaluate on checkpoints start from number ``5000`` to last.

.. code-block::

    python -m lmp.script.evaluate_model_on_dataset wikitext-2 \
        --batch_size 32 \
        --first_ckpt 5000 \
        --exp_name my_exp \
        --ver valid

The following example evaluate on the latest checkpoint.

.. code-block::

    python -m lmp.script.evaluate_model_on_dataset wikitext-2 \
        --batch_size 32 \
        --first_ckpt -1 \
        --exp_name my_exp \
        --ver valid

Specify only some checkpoints to be evaluated.

.. code-block::

    python -m lmp.script.evaluate_model_on_dataset wikitext-2 \
        --batch_size 32 \
        --first_ckpt 5000 \
        --exp_name my_exp \
        --last_ckpt 10000 \
        --ver valid

Since evaluation do not need to calculate backward pass, model will consume
less memory than training, thus we can use larger batch size by increasing
``--batch_size`` to accelerate evaluation process.

.. code-block::

    python -m lmp.script.evaluate_model_on_dataset wikitext-2 \
        --batch_size 128 \
        --ckpt -1 \
        --exp_name my_exp \
        --ver valid
"""

import argparse

import torch
import torch.utils.data
from tqdm import tqdm

import lmp.model
import lmp.util.cfg
import lmp.util.dset
import lmp.util.log
import lmp.util.model
import lmp.util.tknzr
from lmp.dset import DSET_OPTS


def parse_arg() -> argparse.Namespace:
    r"""Parse arguments from CLI.

    Parse pre-trained language model experiment name and evaluate on dataset.
    Argument must begin with a dataset name ``dset_name``.

    --batch_size  Evaluation batch size.
    --ckpt        Pre-trained model checkpoint.
    --exp_name    Pre-trained tokenizer experiment name.
    --ver         Version of the dataset.
                  Default to ``dset``'s default version.

    Returns
    =======
    argparse.Namespace
        Arguments from CLI.
    """
    # Create parser.
    parser = argparse.ArgumentParser(
        'python -m lmp.script.evaluate_model_on_dataset',
        description='Evaluate language model on dataset.',
    )

    # Create subparser for each dataset.
    subparsers = parser.add_subparsers(dest='dset_name', required=True)

    for dset_name, dset_clss in DSET_OPTS.items():
        # Use dataset name as CLI argument.
        dset_parser = subparsers.add_parser(
            dset_name,
            description=f'Evaluate language model on {dset_name} dataset.',
        )

        # Required arguments.
        dset_parser.add_argument(
            '--batch_size',
            help='Evaluation batch size.',
            required=True,
            type=int,
        )
        dset_parser.add_argument(
            '--exp_name',
            help='Pre-trained language model experiment name.',
            required=True,
            type=str,
        )
        dset_parser.add_argument(
            '--first_ckpt',
            help=' '.join([
                'Pre-trained language model first checkpoint to evaluate.',
                'Evaluate all pre-trained language model checkpoints range',
                'from `--first_ckpt` to `--last_ckpt`.',
            ]),
            required=True,
            type=int,
        )

        # Optional arguments.
        dset_parser.add_argument(
            '--last_ckpt',
            default=-1,
            help=' '.join([
                'Pre-trained language model last checkpoint to evaluate.',
                'Evaluate all pre-trained language model checkpoints range',
                'from `--first_ckpt` to `--last_ckpt`.',
                'Last checkpoint is also included to evaluate.',
            ]),
            type=int,
        )
        dset_parser.add_argument(
            '--ver',
            default=None,
            help=' '.join([
                f'Version of the {dset_name} dataset.',
                f'Defaults to {dset_clss.df_ver}.',
            ]),
            choices=dset_clss.vers,
            type=str,
        )

    return parser.parse_args()


def main() -> None:
    r"""Script entry point."""
    # Parse command-line argument.
    args = parse_arg()

    # Get dataset instance with specified version.
    dset = lmp.util.dset.load(dset_name=args.dset_name, ver=args.ver)

    # Mini-batch random sampler.
    dldr = torch.utils.data.DataLoader(
        dataset=dset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Load pre-trained model configuration.
    model_cfg = lmp.util.cfg.load(exp_name=args.exp_name)

    # Load pre-trained tokenizer configuration.
    tknzr_cfg = lmp.util.cfg.load(exp_name=model_cfg.tknzr_exp_name)

    # Load pre-trained tokenizer instance.
    tknzr = lmp.util.tknzr.load(
        exp_name=tknzr_cfg.exp_name,
        tknzr_name=tknzr_cfg.tknzr_name,
    )

    # Get model running device.
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    # Get tensorboard logger instance.
    writer = lmp.util.log.get_tb_logger(exp_name=args.exp_name)

    # Load pre-trained checkpoints range from `args.first_ckpt` to
    # `args.last_ckpt`.
    for ckpt in lmp.util.model.list_ckpts(
        exp_name=args.exp_name,
        first_ckpt=args.first_ckpt,
        last_ckpt=args.last_ckpt,
    ):
        # Load pre-trained model instance from checkpoint `ckpt`.
        model = lmp.util.model.load(
            ckpt=ckpt,
            tknzr=tknzr,
            **model_cfg.__dict__,
        )

        # Set model to evaluation model.
        # This turn off dropout layers in model.
        model = model.eval()

        # Move model to running device.
        model = model.to(device)

        # Record average perplexity.
        avg_ppl = 0.0
        for batch_txt in tqdm(dldr):

            # Encode batch text into batch of token ids.
            batch_tkids = tknzr.batch_enc(
                batch_txt=batch_txt,
                max_seq_len=model_cfg.max_seq_len,
            )

            # Convert batch of token ids to `torch.Tensor` with
            # `dtype == torch.int64`.
            batch_tkids = torch.LongTensor(batch_tkids)

            # Move tensors to model running device.
            batch_tkids = batch_tkids.to(device)

            # Format batch token ids to satisfy language model training format.
            batch_prev_tkids = batch_tkids[..., :-1]
            batch_next_tkids = batch_tkids[..., 1:]

            # Calculate perplexity.
            batch_avg_ppl = model.ppl(
                batch_next_tkids=batch_next_tkids,
                batch_prev_tkids=batch_prev_tkids,
            )

            # Accumulate average perplexity.
            avg_ppl += batch_avg_ppl * len(batch_txt) / len(dset)

        # Log average perplexity on dataset to CLI and tensorboard.
        writer.add_scalar(f'ppl/{args.dset_name}/{args.ver}', avg_ppl, ckpt)
        print(f'checkpoint {ckpt} ppl: {avg_ppl}')


if __name__ == '__main__':
    main()
