"""Test parsing arguments.

Test target:
- :py:meth:`lmp.script.train_tknzr.parse_args`.
"""

import lmp.dset
import lmp.script.train_tknzr
from lmp.tknzr import CharTknzr, WsTknzr


def test_char_tknzr_parse_results(
  exp_name: str,
  is_uncased: bool,
  max_seq_len: int,
  max_vocab: int,
  min_count: int,
  seed: int,
) -> None:
  """Must correctly parse all arguments for :py:class:`lmp.tknzr.CharTknzr`."""
  for dset_name, dset_type in lmp.dset.DSET_OPTS.items():
    for ver in dset_type.vers:
      argv = [
        CharTknzr.tknzr_name,
        '--dset_name',
        dset_name,
        '--exp_name',
        exp_name,
        '--max_seq_len',
        str(max_seq_len),
        '--max_vocab',
        str(max_vocab),
        '--min_count',
        str(min_count),
        '--seed',
        str(seed),
        '--ver',
        ver,
      ]

      if is_uncased:
        argv.append('--is_uncased')

      args = lmp.script.train_tknzr.parse_args(argv=argv)
      assert args.dset_name == dset_name
      assert args.exp_name == exp_name
      assert args.is_uncased == is_uncased
      assert args.max_seq_len == max_seq_len
      assert args.max_vocab == max_vocab
      assert args.min_count == min_count
      assert args.seed == seed
      assert args.tknzr_name == CharTknzr.tknzr_name
      assert args.ver == ver


def test_ws_tknzr_parse_results(
  exp_name: str,
  is_uncased: bool,
  max_seq_len: int,
  max_vocab: int,
  min_count: int,
  seed: int,
) -> None:
  """Must correctly parse all arguments for :py:class:`lmp.tknzr.WsTknzr`."""
  for dset_name, dset_type in lmp.dset.DSET_OPTS.items():
    for ver in dset_type.vers:
      argv = [
        WsTknzr.tknzr_name,
        '--dset_name',
        dset_name,
        '--exp_name',
        exp_name,
        '--max_seq_len',
        str(max_seq_len),
        '--max_vocab',
        str(max_vocab),
        '--min_count',
        str(min_count),
        '--seed',
        str(seed),
        '--ver',
        ver,
      ]

      if is_uncased:
        argv.append('--is_uncased')

      args = lmp.script.train_tknzr.parse_args(argv=argv)
      assert args.dset_name == dset_name
      assert args.exp_name == exp_name
      assert args.is_uncased == is_uncased
      assert args.max_seq_len == max_seq_len
      assert args.max_vocab == max_vocab
      assert args.min_count == min_count
      assert args.seed == seed
      assert args.tknzr_name == WsTknzr.tknzr_name
      assert args.ver == ver
