r"""Evaluate language model on given sample.

Tool for evaluating language model on given sample.
Pre-trained model are used to calculate perplexity on given sample.
This script serve as model validation, and it is usually run after training
model.

See Also
========
lmp.model
    All available models.
lmp.script.train_model
    Train language model.

Examples
========
The following example use ``"Hello world"`` as input sample to evaluate
pre-trained language model experiment ``my_exp``.
It evaluate on the checkpoint number ``5000``.

.. code-block::

    python -m lmp.script.evaluate_model_on_sample \
        --ckpt 5000 \
        --exp_name my_exp \
        --txt "Hello world"

The following example evaluate on the latest checkpoint.

.. code-block::

    python -m lmp.script.evaluate_model_on_sample \
        --ckpt -1 \
        --exp_name my_exp \
        --txt "Hello world"
"""

import argparse

import torch

import lmp.model
import lmp.util.cfg
import lmp.util.model
import lmp.util.tknzr


def parse_arg() -> argparse.Namespace:
    r"""Parse arguments from CLI.

    Parse pre-trained language model experiment name and text to be evaluated.

    --ckpt      Pre-trained model checkpoint.
    --exp_name  Pre-trained tokenizer experiment name.
    --txt       Text to be tokenized.

    Returns
    =======
    argparse.Namespace
        Arguments from CLI.
    """
    # Create parser.
    parser = argparse.ArgumentParser(
        'python -m lmp.script.evaluate_model_on_sample',
        description='Evaluate language model on given sample.',
    )

    # Required arguments.
    parser.add_argument(
        '--ckpt',
        help='Pre-trained language model checkpoint.',
        required=True,
        type=int,
    )
    parser.add_argument(
        '--exp_name',
        help='Pre-trained language model experiment name.',
        required=True,
        type=str,
    )
    parser.add_argument(
        '--txt',
        help='Text to evaluate.',
        required=True,
        type=str,
    )

    return parser.parse_args()


def main() -> None:
    r"""Script entry point."""
    # Parse command-line argument.
    args = parse_arg()

    # Load pre-trained model configuration.
    model_cfg = lmp.util.cfg.load(exp_name=args.exp_name)

    # Load pre-trained tokenizer configuration.
    tknzr_cfg = lmp.util.cfg.load(exp_name=model_cfg.tknzr_exp_name)

    # Load pre-trained tokenizer instance.
    tknzr = lmp.util.tknzr.load(
        exp_name=tknzr_cfg.exp_name,
        tknzr_name=tknzr_cfg.tknzr_name,
    )

    # Load pre-trained model instance.
    model = lmp.util.model.load(
        ckpt=args.ckpt,
        tknzr=tknzr,
        **model_cfg.__dict__,
    )

    # Get model running device.
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    # Set model to evaluation model.
    # This turn off dropout layers in model.
    model = model.eval()

    # Move model to running device.
    model = model.to(device)

    # Encode text into token ids.
    # Wrap as batch with only one sample since `model.ppl` only accept batch.
    batch_tkids = tknzr.batch_enc(
        batch_txt=[args.txt],
        max_seq_len=model_cfg.max_seq_len,
    )

    # Convert token ids to `torch.Tensor` with `dtype == torch.int64`.
    batch_tkids = torch.LongTensor(batch_tkids)

    # Move tensors to model running device.
    batch_tkids = batch_tkids.to(device)

    # Format batch token ids to satisfy language model training format.
    batch_prev_tkids = batch_tkids[..., :-1]
    batch_next_tkids = batch_tkids[..., 1:]

    # Calculate perplexity.
    ppl = model.ppl(
        batch_next_tkids=batch_next_tkids,
        batch_prev_tkids=batch_prev_tkids,
    )

    # Output perplexity on given sample.
    print(ppl)


if __name__ == '__main__':
    main()
