"""Use this script to sample data points of a dataset.

We can use the following script to sample text from :py:class:`~lmp.dset.WikiText2Dset`.

.. code-block:: shell

  python -m lmp.script.sample_dset wiki-text-2

The default sampling index is ``0`` and the default version of :py:class:`~lmp.dset.WikiText2Dset` is ``train``.
Thus the following script has the same sampling result as above.

.. code-block:: shell

  python -m lmp.script.sample_dset wiki-text-2 --idx 0 --ver train

The following script sample text from :py:class:`~lmp.dset.WikiText2Dset` with index set to ``1`` and version set to
``test``.

.. code-block:: shell

  python -m lmp.script.sample_dset wiki-text-2 --idx 1 --ver test

You can use ``-h`` or ``--help`` options to get a list of available datasets.

.. code-block:: shell

  python -m lmp.script.sample_dset -h

You can use ``-h`` or ``--help`` options on a specific dataset to get a list of supported CLI arguments, including all
available versions of a dataset.

.. code-block:: shell

  python -m lmp.script.sample_dset wiki-text-2 -h

See Also
--------
:doc:`lmp.dset </dset/index>`
  All available datasets.
"""

import argparse
import sys
from typing import List

import lmp.dset
import lmp.util.dset
import lmp.util.rand
import lmp.util.validate


def parse_args(argv: List[str]) -> argparse.Namespace:
  """Parse CLI arguments.

  Parameters
  ----------
  argv: list[str]
    List of CLI arguments.

  See Also
  --------
  sys.argv
    Python CLI arguments interface.

  Returns
  -------
  argparse.Namespace
    Parsed CLI arguments.
  """
  # Create parser.
  parser = argparse.ArgumentParser('python -m lmp.script.sample_dset', description='Sample dataset.')

  # Use dataset name to create subparser for all datasets.
  subparsers = parser.add_subparsers(dest='dset_name', required=True)
  for dset_name, dset_type in lmp.dset.DSET_OPTS.items():
    dset_subparser = subparsers.add_parser(dset_name, description=f'Sample ``lmp.dset.{dset_type.__name__}``.')

    dset_subparser.add_argument(
      '--idx',
      default=0,
      help='''
      Index of targeting sample.
      Default is ``0``.
      ''',
      type=int,
    )
    dset_subparser.add_argument(
      '--seed',
      default=42,
      help='''
      Random seed.
      Default is ``42``.
      ''',
      type=int,
    )
    dset_subparser.add_argument(
      '--ver',
      default=dset_type.df_ver,
      help=f'''
      Dataset version of ``lmp.dset.{dset_type.__name__}``.
      Default version is ``{dset_type.df_ver}``.
      ''',
      choices=dset_type.vers,
      type=str,
    )

  args = parser.parse_args(argv)

  # `args.idx` validation.
  lmp.util.validate.raise_if_wrong_ordered(vals=[0, args.idx], val_names=['0', 'args.idx'])

  return args


def main(argv: List[str]) -> None:
  """Script entry point.

  Parameters
  ----------
  argv: list[str]
    List of CLI arguments.

  Returns
  -------
  None
  """
  # Parse CLI arguments.
  args = parse_args(argv=argv)

  # Set random seed for reproducibility.
  lmp.util.rand.set_seed(seed=args.seed)

  # Get dataset instance with specified version.
  dset = lmp.util.dset.load(**args.__dict__)

  # Output sample result.
  print(dset[args.idx])


if __name__ == '__main__':
  main(argv=sys.argv[1:])
