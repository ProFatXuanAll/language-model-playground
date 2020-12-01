r"""Train tokenizer.

Tool for training tokenizer on particular dataset.
This script is usually run before training model.

See Also
========
lmp.script.tokenize
    Use pre-trained tokenizer to tokenize text.
lmp.tknzr
    All available tokenizers.

Examples
========
The following example train :py:class:`lmp.tknzr.WsTknzr` (``whitespace``)
on :py:class:`lmp.dset.WikiText2Dset` using ``train`` version.
(``--dset_name wikitext-2`` and ``--ver train``).

.. code-block:: sh

    python -m lmp.script.train_tokenizer whitespace \
        --dset_name wikitext-2 \
        --exp_name my_exp \
        --max_vocab 10 \
        --min_count 2 \
        --ver train

The training result will be save at ``exp/my_exp``, and can be reused by other
scripts.

One can include more tokens in vocabulary using ``--max_vocab``:

.. code-block:: sh

    python -m lmp.script.train_tokenizer whitespace \
        --dset_name wikitext-2 \
        --exp_name my_exp \
        --max_vocab 10000 \
        --min_count 2 \
        --ver train

Set ``--max_vocab`` to ``-1`` to include all tokens in the dataset:

.. code-block:: sh

    python -m lmp.script.train_tokenizer whitespace \
        --dset_name wikitext-2 \
        --exp_name my_exp \
        --max_vocab -1 \
        --min_count 2 \
        --ver train

Use ``--min_count`` to filter out tokens such as typos, names, locations, etc.

.. code-block:: sh

    python -m lmp.script.train_tokenizer whitespace
        --dset_name wikitext-2 \
        --exp_name my_exp \
        --max_vocab 10000 \
        --min_count 5 \
        --ver train

Use ``--is_uncased`` to avoid differ tokens with same charaters but in
different case.

.. code-block:: sh

    python -m lmp.script.train_tokenizer whitespace
        --dset_name wikitext-2 \
        --exp_name my_exp \
        --is_uncased \
        --max_vocab 10000 \
        --min_count 5 \
        --ver train

Use ``-h`` or ``--help`` options to get list of available tokenizers.

.. code-block:: sh

    python -m lmp.script.train_tokenizer -h
"""

import argparse

import lmp.dset
import lmp.tknzr
import lmp.util.cfg
import lmp.util.dset
import lmp.util.tknzr


def parse_arg() -> argparse.Namespace:
    r"""Parse arguments from CLI.

    Argument must begin with a tokenizer name ``tknzr_name``.
    All arguments are added with tokenizer's static method ``train_parser``.

    Returns
    =======
    argparse.Namespace
        Arguments from CLI.
    """
    # Create parser.
    parser = argparse.ArgumentParser(
        'python -m lmp.script.train_tokenizer',
        description='Train tokenizer.',
    )

    # Create subparser for each tokenizer.
    subparsers = parser.add_subparsers(dest='tknzr_name', required=True)

    for tknzr_name, tknzr_clss in lmp.tknzr.TKNZR_OPTS.items():
        # Use tokenizer name as CLI argument.
        tknzr_parser = subparsers.add_parser(
            tknzr_name,
            description=f'Training {tknzr_name} tokenizer.',
        )

        # Add customized arguments.
        tknzr_clss.train_parser(tknzr_parser)

    return parser.parse_args()


def main() -> None:
    r"""Script entry point."""
    # Parse command-line argument.
    args = parse_arg()

    # Save training configuration.
    lmp.util.cfg.save(args=args, exp_name=args.exp_name)

    # Get dataset instance with specified version.
    dset = lmp.util.dset.load(dset_name=args.dset_name, ver=args.ver)

    # Get new tokenizer instance.
    tknzr = lmp.util.tknzr.create(**args.__dict__)

    # Build tokenizer's vocabulary.
    tknzr.build_vocab(dset)

    # Save training result.
    tknzr.save(args.exp_name)


if __name__ == '__main__':
    main()
