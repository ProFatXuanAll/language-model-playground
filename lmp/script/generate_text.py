r"""Generate text using pre-trained language model.

Automatically generate text with pred-trained language model condition on given
text segment.
Pre-trained model are used to generate text.
This script serve as model validation, and it is usually run after training
model.

See Also
========
lmp.model
    All available models.
lmp.infer
    All available inference methods.
lmp.script.train_model
    Train language model.

Examples
========
The following example use ``"Hello world"`` as conditional text segment to
generate text with pre-trained language model experiment ``my_exp``.
It use ``top-1`` as inference method to generate text.
It load pre-trained language model on checkpoint number ``5000``.

.. code-block::

    python -m lmp.script.generate_text top-1 \
        --ckpt 5000 \
        --exp_name my_exp \
        --txt "Hello world"
"""

import argparse

import torch

import lmp.model
import lmp.util.cfg
import lmp.util.infer
import lmp.util.model
import lmp.util.rand
import lmp.util.tknzr
from lmp.infer import INFER_OPTS


def parse_arg() -> argparse.Namespace:
    r"""Parse arguments from CLI.

    Argument must begin with a inference method name ``infer_name``.
    All arguments are added with inference method's static method
    ``infer_parser``.

    Returns
    =======
    argparse.Namespace
        Arguments from CLI.
    """
    # Create parser.
    parser = argparse.ArgumentParser(
        'python -m lmp.script.generate_text',
        description='Generate text using language model.',
    )

    # Create subparser for each inference method.
    subparsers = parser.add_subparsers(dest='infer_name', required=True)

    for infer_name, infer_clss in INFER_OPTS.items():
        # Use dataset name as CLI argument.
        infer_parser = subparsers.add_parser(
            infer_name,
            description=f'Use {infer_name} as inference method.',
        )

        # Add customized arguments.
        infer_clss.infer_parser(infer_parser)

    return parser.parse_args()


def main() -> None:
    r"""Script entry point."""
    # Parse command-line argument.
    args = parse_arg()

    # Set random seed for reproducibility.
    lmp.util.rand.set_seed(seed=args.seed)

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

    # Get inference method.
    infer = lmp.util.infer.create(
        max_seq_len=model_cfg.max_seq_len,
        **args.__dict__,
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

    # Generate text with specified inference method.
    txt = infer.gen(model=model, tknzr=tknzr, txt=args.txt)

    # Output generate text.
    print(txt)


if __name__ == '__main__':
    main()
