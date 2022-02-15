"""Test adding CLI arguments to parser.

Test target:
- :py:meth:`lmp.tknzr._ws.WsTknzr.add_CLI_args`.
"""

import argparse

from lmp.tknzr._ws import WsTknzr


def test_arguments(max_seq_len: int, max_vocab: int, min_count: int) -> None:
  """Must have correct arguments."""
  parser = argparse.ArgumentParser()
  WsTknzr.add_CLI_args(parser=parser)
  args = parser.parse_args(
    [
      '--max_seq_len',
      str(max_seq_len),
      '--max_vocab',
      str(max_vocab),
      '--min_count',
      str(min_count),
    ]
  )
  assert not args.is_uncased
  assert args.max_seq_len == max_seq_len
  assert args.max_vocab == max_vocab
  assert args.min_count == min_count
  args = parser.parse_args(
    [
      '--is_uncased',
      '--max_seq_len',
      str(max_seq_len),
      '--max_vocab',
      str(max_vocab),
      '--min_count',
      str(min_count),
    ]
  )
  assert args.is_uncased
  assert args.max_seq_len == max_seq_len
  assert args.max_vocab == max_vocab
  assert args.min_count == min_count
