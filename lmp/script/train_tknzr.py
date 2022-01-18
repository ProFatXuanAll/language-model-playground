r"""Train tokenizer.

Use this script to train tokenizer on particular dataset.  This script is usually run before training model.

See Also
--------
lmp.dset
  All available datasets.
lmp.script.sample_dset
  Get a glimpse on all available datasets.
lmp.script.tokenize
  Use pre-trained tokenizer to perform tokenization on given text.
lmp.tknzr
  All available tokenizers.

Examples
--------
The following example script train a whitespace tokenizer :py:class:`lmp.tknzr.WsTknzr` on Wiki-Text-2 dataset
:py:class:`lmp.dset.WikiText2Dset` with ``train`` version.

.. code-block:: shell

   python -m lmp.script.train_tknzr whitespace \
     --dset_name wiki-text-2 \
     --exp_name my_exp \
     --max_vocab 10 \
     --min_count 2 \
     --ver train

The training result will be save at path ``root/exp/my_exp`` and can be reused by other scripts.  Here ``root`` refers
to :py:attr:`lmp.util.path.PROJECT_ROOT`.

One can increase ``--max_vocab`` to allow tokenizer to include more tokens into its vocabulary:

.. code-block:: shell

   python -m lmp.script.train_tknzr whitespace \
     --dset_name wiki-text-2 \
     --exp_name my_exp \
     --max_vocab 10000 \
     --min_count 2 \
     --ver train

Set ``--max_vocab`` to ``-1`` to include all tokens in :py:class:`lmp.dset.WikiText2Dset` into tokenizer's vocabulary:

.. code-block:: shell

   python -m lmp.script.train_tknzr whitespace \
     --dset_name wiki-text-2 \
     --exp_name my_exp \
     --max_vocab -1 \
     --min_count 2 \
     --ver train

Tokens have low occurrence counts may indicate typos, named entities (people, locations, organizations, etc.) or random
character combinations (emojis, glyphs, etc.).  Sometimes one does not want to include tokens have low occurrence
counts.  Use ``--min_count`` to filter out tokens have occurrence counts lower than ``--min_count``.

.. code-block:: shell

   python -m lmp.script.train_tknzr whitespace \
     --dset_name wiki-text-2 \
     --exp_name my_exp \
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
     --exp_name my_exp \
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
import sys
from typing import List

import lmp.dset
import lmp.tknzr
import lmp.util.cfg
import lmp.util.dset
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
    tknzr_parser = subparsers.add_parser(tknzr_name, description=f'Training {tknzr_name} tokenizer.')

    # Add tokenizer specific arguments.
    tknzr_type.train_parser(tknzr_parser)

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

  # Get dataset instance with specified version.
  dset = lmp.util.dset.load(dset_name=args.dset_name, ver=args.ver)

  # Get new tokenizer instance.
  tknzr = lmp.util.tknzr.create(**args.__dict__)

  # Build tokenizer's vocabulary.
  tknzr.build_vocab(batch_txt=dset)

  # Save training result.
  tknzr.save(exp_name=args.exp_name)


if __name__ == '__main__':
  main(argv=sys.argv[1:])
