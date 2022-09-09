"""Test adding CLI arguments to parser.

Test target:
- :py:meth:`lmp.tknzr._ws.WsTknzr.add_CLI_args`.
"""

import argparse

from lmp.tknzr._ws import WsTknzr


def test_default_value() -> None:
  """Ensure default value consistency."""
  parser = argparse.ArgumentParser()
  WsTknzr.add_CLI_args(parser=parser)
  args = parser.parse_args([])

  assert not args.is_uncased
  assert args.max_vocab == -1
  assert args.min_count == 0


def test_arguments(is_uncased: bool, max_vocab: int, min_count: int) -> None:
  """Must have correct arguments."""
  parser = argparse.ArgumentParser()
  WsTknzr.add_CLI_args(parser=parser)
  argv = [
    '--max_vocab',
    str(max_vocab),
    '--min_count',
    str(min_count),
  ]

  if is_uncased:
    argv.append('--is_uncased')

  args = parser.parse_args(argv)

  assert args.is_uncased == is_uncased
  assert args.max_vocab == max_vocab
  assert args.min_count == min_count
