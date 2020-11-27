r"""Use pre-trained tokenizer to tokenize text.

Pre-trained tokenizer must exist, i.e., perform tokenizer training first then
use this script.

See Also
========
lmp.script.train_tokenizer
    Train tokenizer.
lmp.tknzr
    All available tokenizers.

Examples
========
The following example used pre-trained tokenizer under experiment ``my_exp`` to
tokenize text ``'hello world'``.

.. code-block:: sh

    python -m lmp.script.tokenize \
        --exp_name my_exp \
        --txt "Hello World"

Use ``-h`` or ``--help`` options to get list of available options.

.. code-block:: sh

    python -m lmp.script.train_tokenizer -h
"""

import argparse

import lmp.dset
import lmp.tknzr
import lmp.util.cfg
import lmp.util.tknzr


def parse_arg() -> argparse.Namespace:
    r"""Parse arguments from CLI.

    Parse pre-trained tokenizer experiment name and text to be tokenized.

    --exp_name  Pre-trained tokenizer experiment name.
    --txt       Text to be tokenized.

    Returns
    =======
    argparse.Namespace
        Arguments from CLI.
    """
    # Create parser.
    parser = argparse.ArgumentParser(
        'python -m lmp.script.tokenize',
        description='Use pre-trained tokenizer to tokenize text.',
    )

    # Required arguments.
    parser.add_argument(
        '--exp_name',
        help='Pre-trained tokenizer experiment name.',
        required=True,
        type=str,
    )
    parser.add_argument(
        '--txt',
        help='Text to be tokenized.',
        required=True,
        type=str,
    )

    return parser.parse_args()


def main() -> None:
    r"""Script entry point."""
    # Parse command-line argument.
    args = parse_arg()

    # Load pre-trained tokenizer configuration.
    tknzr_cfg = lmp.util.cfg.load(exp_name=args.exp_name)

    # Load pre-trained tokenizer instance.
    tknzr = lmp.util.tknzr.load(
        exp_name=args.exp_name,
        tknzr_name=tknzr_cfg.tknzr_name,
    )

    # Tokenize text.
    print(tknzr.tknz(args.txt))


if __name__ == '__main__':
    main()
