r"""Use pre-trained tokenizer to tokenize text.

One must first run the script :doc:`lmp.script.train_tknzr </script/train_tknzr>` before running this script.

The following example used pre-trained tokenizer under experiment ``my_tknzr_exp`` to tokenize text ``'Hello World'``.

.. code-block:: shell

  python -m lmp.script.tknz_txt --exp_name my_tknzr_exp --txt "Hello World"

You can use ``-h`` or ``--help`` options to get a list of supported CLI arguments.

.. code-block:: shell

  python -m lmp.script.tknz_txt -h

See Also
--------
:doc:`lmp.script.train_tknzr </script/train_tknzr>`
  Train tokenizer.
:doc:`lmp.tknzr </tknzr/index>`
  All available tokenizers.
"""

import argparse
import sys
from typing import List

import lmp.dset
import lmp.tknzr
import lmp.util.rand
import lmp.util.tknzr
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
  parser = argparse.ArgumentParser(
    'python -m lmp.script.tknz_txt',
    description='Use pre-trained tokenizer to tokenize text.',
  )

  # Required arguments.
  parser.add_argument(
    '--exp_name',
    default='my_tknzr_exp',
    help='''
    Pre-trained tokenizer experiment name.
    Default is ``my_tknzr_exp``.
    ''',
    type=str,
  )
  parser.add_argument(
    '--seed',
    default=42,
    help='''
    Random seed.
    Default is ``42``.
    ''',
    type=int,
  )
  parser.add_argument(
    '--txt',
    default='',
    help='''
    Text to be tokenized.
    Default is empty string.
    ''',
    type=str,
  )

  args = parser.parse_args(argv)

  # `args.txt` validation.
  lmp.util.validate.raise_if_not_instance(val=args.txt, val_name='args.txt', val_type=str)

  return args


def main(argv: List[str]) -> List[str]:
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

  # Load pre-trained tokenizer instance.
  tknzr = lmp.util.tknzr.load(exp_name=args.exp_name)

  # Tokenize text.
  return tknzr.tknz(args.txt)


if __name__ == '__main__':
  tks = main(argv=sys.argv[1:])
  print(tks)
