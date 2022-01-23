"""Test parser arguments.

Test target:
- :py:meth:`lmp.tknzr.BaseTknzr.train_parser`.
"""

import argparse

import lmp.dset
from lmp.tknzr import BaseTknzr


def test_arguments(exp_name: str, max_vocab: int, min_count: int, seed: int) -> None:
  """Must have correct arguments."""
  parser = argparse.ArgumentParser()
  BaseTknzr.train_parser(parser=parser)
  for dset_name, dset_type in lmp.dset.DSET_OPTS.items():
    for ver in dset_type.vers:
      args = parser.parse_args(
        [
          '--dset_name',
          dset_name,
          '--exp_name',
          exp_name,
          '--max_vocab',
          str(max_vocab),
          '--min_count',
          str(min_count),
          '--seed',
          str(seed),
          '--ver',
          ver,
        ]
      )
      assert args.dset_name == dset_name
      assert args.exp_name == exp_name
      assert not args.is_uncased
      assert args.max_vocab == max_vocab
      assert args.min_count == min_count
      assert args.seed == seed
      assert args.ver == ver
      args = parser.parse_args(
        [
          '--dset_name',
          dset_type.dset_name,
          '--exp_name',
          exp_name,
          '--is_uncased',
          '--max_vocab',
          str(max_vocab),
          '--min_count',
          str(min_count),
          '--seed',
          str(seed),
          '--ver',
          ver,
        ]
      )
      assert args.dset_name == dset_name
      assert args.exp_name == exp_name
      assert args.is_uncased
      assert args.max_vocab == max_vocab
      assert args.min_count == min_count
      assert args.seed == seed
      assert args.ver == ver
