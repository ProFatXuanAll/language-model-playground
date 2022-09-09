"""Test adding CLI arguments to parser.

Test target:
- :py:meth:`lmp.tknzr._bpe.BPETknzr.add_CLI_args`.
"""

import argparse

from lmp.tknzr._bpe import BPETknzr


def test_default_value() -> None:
  """Ensure default value consistency."""
  parser = argparse.ArgumentParser()
  BPETknzr.add_CLI_args(parser=parser)
  args = parser.parse_args([])

  assert not args.is_uncased
  assert args.max_vocab == -1
  assert args.min_count == 0
  assert args.n_merge == 10000


def test_arguments(is_uncased: bool, max_vocab: int, min_count: int, n_merge: int) -> None:
  """Must have correct arguments."""
  parser = argparse.ArgumentParser()
  BPETknzr.add_CLI_args(parser=parser)
  argv = [
    '--max_vocab',
    str(max_vocab),
    '--min_count',
    str(min_count),
    '--n_merge',
    str(n_merge),
  ]

  if is_uncased:
    argv.append('--is_uncased')

  args = parser.parse_args(argv)

  assert args.is_uncased == is_uncased
  assert args.max_vocab == max_vocab
  assert args.min_count == min_count
  assert args.n_merge == n_merge
