r"""Sample dataset using index.

Tool for observing data point in specified dataset.
Use index to sample from dataset.

See Also
========
lmp.dset
    All available dataset.

Examples
========
The following example sample index ``0`` from
:py:class:`lmp.dset.WikiText2Dset` ``train`` dataset.

.. code-block:: sh

    python -m lmp.script.sample_from_dataset wikitext-2

The following example sample index ``1`` from
:py:class:`lmp.dset.WikiText2Dset` ``train`` dataset.

.. code-block:: sh

    python -m lmp.script.sample_from_dataset wikitext-2 --idx 1


The following example sample index ``1`` from
:py:class:`lmp.dset.WikiText2Dset` ``test`` dataset.

.. code-block:: sh

    python -m lmp.script.sample_from_dataset wikitext-2 --idx 1 --ver test

Use ``-h`` or ``--help`` options to get list of available dataset.

.. code-block:: sh

    python -m lmp.script.sample_from_dataset -h

Use ``-h`` or ``--help`` options on specific dataset to get a list of available
versions.

.. code-block:: sh

    python -m lmp.script.sample_from_dataset wikitext-2 -h
"""

import argparse

import lmp.util.dset
from lmp.dset import DSET_OPTS


def parse_arg() -> argparse.Namespace:
    r"""Parse arguments from CLI.

    Argument must begin with a dataset name ``dset_name``.
    The following arguments are optional:

    --ver  Version of the dataset.
           Default to ``dset``'s default version.
    --idx  Sample index.
           Default to ``0``.

    Returns
    =======
    argparse.Namespace
        Arguments from CLI.
    """
    # Create parser.
    parser = argparse.ArgumentParser(
        'python -m lmp.script.sample_from_dataset',
        description='Sample dataset using index.',
    )

    # Create subparser for each dataset.
    subparsers = parser.add_subparsers(dest='dset_name', required=True)

    for dset_name, dset_clss in DSET_OPTS.items():
        # Use dataset name as CLI argument.
        dset_parser = subparsers.add_parser(
            dset_name,
            description=f'Sample {dset_name} dataset using index.',
        )

        # Optional arguments.
        dset_parser.add_argument(
            '--idx',
            default=0,
            help='Sample index.',
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

    # Output sample result.
    print(dset[args.idx])


if __name__ == '__main__':
    main()
