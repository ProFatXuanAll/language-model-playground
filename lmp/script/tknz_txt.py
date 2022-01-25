r"""Use pre-trained tokenizer to tokenize text.

One must first run the script :py:mod:`lmp.script.train_tknzr` before running this script.

See Also
--------
lmp.script.train_tknzr
  Train tokenizer.
lmp.tknzr
  All available tokenizers.

Examples
--------
The following example used pre-trained tokenizer under experiment ``my_tknzr_exp`` to tokenize text ``'hello world'``.

.. code-block:: shell

   python -m lmp.script.tknz_txt \
     --exp_name my_tknzr_exp \
     --txt "Hello World"

You can use ``-h`` or ``--help`` options to get a list of supported CLI arguments.

.. code-block:: shell

   python -m lmp.script.tknz_txt -h
"""

import argparse
import gc
import sys
from typing import List

import lmp.dset
import lmp.tknzr
import lmp.util.cfg
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
  parser = argparse.ArgumentParser(
    'python -m lmp.script.tknz_txt',
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

  # Optional arguments.
  parser.add_argument(
    '--seed',
    default=42,
    help='Random seed.',
    type=int,
  )

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

  # Set random seed for reproducibility.
  lmp.util.rand.set_seed(seed=args.seed)

  # Load pre-trained tokenizer configuration.
  tknzr_cfg = lmp.util.cfg.load(exp_name=args.exp_name)

  # Load pre-trained tokenizer instance.
  tknzr = lmp.util.tknzr.load(exp_name=args.exp_name, tknzr_name=tknzr_cfg.tknzr_name)

  # Tokenize text.
  print(tknzr.tknz(args.txt))

  # Free memory.  This is only need for unit test.
  del args
  del tknzr
  del tknzr_cfg
  gc.collect()


if __name__ == '__main__':
  main(argv=sys.argv[1:])
