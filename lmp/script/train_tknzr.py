r"""Use this script to train tokenizer on a dataset.

This script is usually run before training language model.

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

Examples
--------
The following example script train a whitespace tokenizer :py:class:`lmp.tknzr.WsTknzr` on Wiki-Text-2 dataset
:py:class:`lmp.dset.WikiText2Dset` with ``train`` version.

.. code-block:: shell

   python -m lmp.script.train_tknzr whitespace \
     --dset_name wiki-text-2 \
     --exp_name my_tknzr_exp \
     --max_vocab 10 \
     --min_count 2 \
     --ver train

The training result will be saved at path ``project_root/exp/my_tknzr_exp`` and can be reused by other scripts.

One can increase ``--max_vocab`` to allow tokenizer to include more tokens into its vocabulary:

.. code-block:: shell

   python -m lmp.script.train_tknzr whitespace \
     --dset_name wiki-text-2 \
     --exp_name my_tknzr_exp \
     --max_vocab 10000 \
     --min_count 2 \
     --ver train

Set ``--max_vocab`` to ``-1`` to include all tokens in :py:class:`lmp.dset.WikiText2Dset` into tokenizer's vocabulary:

.. code-block:: shell

   python -m lmp.script.train_tknzr whitespace \
     --dset_name wiki-text-2 \
     --exp_name my_tknzr_exp \
     --max_vocab -1 \
     --min_count 2 \
     --ver train

Tokens have low occurrence counts may indicate typos, named entities (people, locations, organizations, etc.) or random
character combinations (emojis, glyphs, etc.).  Sometimes one does not want to include tokens have low occurrence
counts.  Use ``--min_count`` to filter out tokens have occurrence counts lower than ``--min_count``.

.. code-block:: shell

   python -m lmp.script.train_tknzr whitespace \
     --dset_name wiki-text-2 \
     --exp_name my_tknzr_exp \
     --max_vocab 10000 \
     --min_count 5 \
     --ver train

Sometimes cases do not matter, sometimes they do matter.  For example:

  I ate an apple.
  Apple is a fruit.
  Apple is a company.

The words `apple` and `Apple` in the first two sentences have the meaning of edible fruit regardless of `apple` being
upper case `Apple` or lower case `apple`.  But in the third sentence the word `Apple` has the meaning of smartphone
company and can only be upper case (which represent the name of an entity).  Thus when processing text one must decide
whether to treat cases as a whole or differently.  In this script one can use ``--is_uncased`` to treat upper cases as
same as lower cases.

.. code-block:: shell

   python -m lmp.script.train_tknzr whitespace
     --dset_name wiki-text-2 \
     --exp_name my_tknzr_exp \
     --is_uncased \
     --max_vocab 10000 \
     --min_count 5 \
     --ver train

You can use ``-h`` or ``--help`` options to get a list of available tokenizers.

.. code-block:: shell

   python -m lmp.script.train_tknzr -h

You can use ``-h`` or ``--help`` options on a specific tokenizer to get a list of supported CLI arguments.

.. code-block:: shell

   python -m lmp.script.train_tknzr whitespace -h
"""

import argparse
import gc
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
    group = tknzr_subparser.add_argument_group('tokenizer training arguments')
    group.add_argument(
      '--dset_name',
      choices=lmp.dset.DSET_OPTS.keys(),
      help='Name of the dataset which will be used to train tokenizer.',
      required=True,
      type=str,
    )
    group.add_argument(
      '--exp_name',
      help='Name of the tokenizer training experiment.',
      required=True,
      type=str,
    )
    group.add_argument(
      '--ver',
      help='Version of the dataset.',
      required=True,
      type=str,
    )

    # Optional arguments.
    group.add_argument(
      '--seed',
      default=42,
      help='Random seed.  Default is ``42``.',
      type=int,
    )

    # Add tokenizer specific arguments.
    tknzr_type.add_CLI_args(parser=tknzr_subparser)

  return parser.parse_args(argv)


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

  # Free memory.  This is only need for unit test.
  del args
  del dset
  del tknzr
  gc.collect()


if __name__ == '__main__':
  main(argv=sys.argv[1:])
