"""Test parsing arguments.

Test target:
- :py:meth:`lmp.script.train_tknzr.parse_args`.
"""

import lmp.script.train_tknzr
from lmp.dset import ALL_DSETS
from lmp.tknzr import CharTknzr, WsTknzr


def test_char_tknzr_parse_results(exp_name: str, max_vocab: int, min_count: int) -> None:
  """Must correctly parse all arguments for :py:class:`lmp.tknzr.CharTknzr`."""
  for dset_type in ALL_DSETS:
    for ver in dset_type.vers:
      # Case-sensitive.
      args = lmp.script.train_tknzr.parse_args(
        argv=filter(
          bool, [
            CharTknzr.tknzr_name,
            '--dset_name',
            dset_type.dset_name,
            '--exp_name',
            exp_name,
            '--max_vocab',
            str(max_vocab),
            '--min_count',
            str(min_count),
            '--ver',
            ver,
          ]
        )
      )
      assert args.dset_name == dset_type.dset_name
      assert args.exp_name == exp_name
      assert not args.is_uncased
      assert args.max_vocab == max_vocab
      assert args.min_count == min_count
      assert args.tknzr_name == CharTknzr.tknzr_name
      assert args.ver == ver
      # Case-insensitive.
      args = lmp.script.train_tknzr.parse_args(
        argv=filter(
          bool, [
            CharTknzr.tknzr_name,
            '--dset_name',
            dset_type.dset_name,
            '--exp_name',
            exp_name,
            '--is_uncased',
            '--max_vocab',
            str(max_vocab),
            '--min_count',
            str(min_count),
            '--ver',
            ver,
          ]
        )
      )
      assert args.dset_name == dset_type.dset_name
      assert args.exp_name == exp_name
      assert args.is_uncased
      assert args.max_vocab == max_vocab
      assert args.min_count == min_count
      assert args.tknzr_name == CharTknzr.tknzr_name
      assert args.ver == ver


def test_ws_tknzr_parse_results(exp_name: str, max_vocab: int, min_count: int) -> None:
  """Must correctly parse all arguments for :py:class:`lmp.tknzr.WsTknzr`."""
  for dset_type in ALL_DSETS:
    for ver in dset_type.vers:
      # Case-sensitive.
      args = lmp.script.train_tknzr.parse_args(
        argv=filter(
          bool, [
            WsTknzr.tknzr_name,
            '--dset_name',
            dset_type.dset_name,
            '--exp_name',
            exp_name,
            '--max_vocab',
            str(max_vocab),
            '--min_count',
            str(min_count),
            '--ver',
            ver,
          ]
        )
      )
      assert args.dset_name == dset_type.dset_name
      assert args.exp_name == exp_name
      assert not args.is_uncased
      assert args.max_vocab == max_vocab
      assert args.min_count == min_count
      assert args.tknzr_name == WsTknzr.tknzr_name
      assert args.ver == ver
      # Case-insensitive.
      args = lmp.script.train_tknzr.parse_args(
        argv=filter(
          bool, [
            WsTknzr.tknzr_name,
            '--dset_name',
            dset_type.dset_name,
            '--exp_name',
            exp_name,
            '--is_uncased',
            '--max_vocab',
            str(max_vocab),
            '--min_count',
            str(min_count),
            '--ver',
            ver,
          ]
        )
      )
      assert args.dset_name == dset_type.dset_name
      assert args.exp_name == exp_name
      assert args.is_uncased
      assert args.max_vocab == max_vocab
      assert args.min_count == min_count
      assert args.tknzr_name == WsTknzr.tknzr_name
      assert args.ver == ver
