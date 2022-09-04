r"""Use this script to train tokenizer on a dataset.

This script must be run before training language model.
Once a tokenizer is trained it can be shared throughout different scripts.

The following script train a character tokenizer :py:class:`~lmp.tknzr.CharTknzr` on demo dataset
:py:class:`~lmp.dset.DemoDset`.

.. code-block:: shell

  python -m lmp.script.train_tknzr character

The tokenizer training experiment is named as ``my_tknzr_exp``.
The training result will be saved at path ``project_root/exp/my_tknzr_exp`` and can be reused by other scripts.
To use different name, one can set the ``--exp_name`` argument.

.. code-block:: shell

  python -m lmp.script.train_tknzr character --exp_name other_name

One might need to train tokenizer on different dataset.
This can be achieved using ``--dset_name`` argument.

.. code-block:: shell

  python -m lmp.script.train_tknzr character --dset_name wiki-text-2

Tokenizer's hyperparameters can be passed as arguments.
For example, one can set ``max_vocab`` and ``min_count`` using ``--max_vocab`` and ``--min_count`` arguments.

.. code-block:: shell

  python -m lmp.script.train_tknzr character --max_vocab 100 --min_count 2

Note that boolean hyperparameters are set to ``False`` if not given, and set to ``True`` if given.

.. code-block:: shell

  # Setting `is_uncased=False`.
  python -m lmp.script.train_tknzr character
  # Setting `is_uncased=True`.
  python -m lmp.script.train_tknzr character --is_uncased

To train a different tokenizer, change the first argument to the specific tokenizer's name.

.. code-block:: shell

  python -m lmp.script.train_tknzr whitespace

You can use ``-h`` or ``--help`` options to get a list of available tokenizers.

.. code-block:: shell

  python -m lmp.script.train_tknzr -h

You can use ``-h`` or ``--help`` options on a specific tokenizer to get a list of supported CLI arguments.

.. code-block:: shell

  python -m lmp.script.train_tknzr whitespace -h

See Also
--------
:doc:`lmp.dset </dset/index>`
  All available datasets.
:doc:`lmp.script.sample_dset </script/sample_dset>`
  Get a glimpse on all available datasets.
:doc:`lmp.script.tknz_txt </script/tknz_txt>`
  Use pre-trained tokenizer to perform tokenization on given text.
:doc:`lmp.tknzr </tknzr/index>`
  All available tokenizers.
"""

import argparse
import sys
from typing import List

import lmp.dset
import lmp.tknzr
import lmp.util.cfg
import lmp.util.dset
import lmp.util.rand
import lmp.util.tknzr


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
  parser = argparse.ArgumentParser('python -m lmp.script.train_tknzr', description='Train tokenizer.')

  # Use tokenizer name to create subparser for all tokenizers.
  subparsers = parser.add_subparsers(dest='tknzr_name', required=True)
  for tknzr_name, tknzr_type in lmp.tknzr.TKNZR_OPTS.items():
    tknzr_subparser = subparsers.add_parser(
      tknzr_name,
      description=f'Training `lmp.tknzr.{tknzr_type.__name__}` tokenizer.',
    )

    # Required arguments.
    group = tknzr_subparser.add_argument_group('tokenizer training hyperparameters')
    group.add_argument(
      '--dset_name',
      choices=lmp.dset.DSET_OPTS.keys(),
      default=lmp.dset.DemoDset.dset_name,
      help=f'''
      Name of the dataset which will be used to train tokenizer.
      Default is ``{lmp.dset.DemoDset.dset_name}``.
      ''',
      type=str,
    )
    group.add_argument(
      '--exp_name',
      default='my_tknzr_exp',
      help='''
      Name of the tokenizer training experiment.
      Default is ``my_tknzr_exp``.
      ''',
      type=str,
    )
    group.add_argument(
      '--seed',
      default=42,
      help='''
      Random seed.
      Default is ``42``.
      ''',
      type=int,
    )
    group.add_argument(
      '--ver',
      default=None,
      help='''
      Version of the dataset.
      Default is ``None``.
      ''',
      type=str,
    )

    # Add tokenizer specific arguments.
    tknzr_type.add_CLI_args(parser=tknzr_subparser)

  args = parser.parse_args(argv)

  if args.ver is None:
    args.ver = lmp.util.dset.DSET_OPTS[args.dset_name].df_ver

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

  # Save training configuration.
  lmp.util.cfg.save(args=args, exp_name=args.exp_name)

  # Set random seed for reproducibility.
  lmp.util.rand.set_seed(seed=args.seed)

  # Get dataset instance with specified version.
  dset = lmp.util.dset.load(**args.__dict__)

  # Get new tokenizer instance.
  tknzr = lmp.util.tknzr.create(**args.__dict__)

  # Build tokenizer's vocabulary.
  tknzr.build_vocab(batch_txt=dset)

  # Save training result.
  lmp.util.tknzr.save(exp_name=args.exp_name, tknzr=tknzr)


if __name__ == '__main__':
  main(argv=sys.argv[1:])
