"""Test parsing :py:class:`lmp.tknzr.BPETknzr` arguments.

Test target:
- :py:meth:`lmp.script.train_tknzr.parse_args`.
- :py:meth:`lmp.tknzr.BPETknzr.add_CLI_args`.
"""

import lmp.dset
import lmp.script.train_tknzr
from lmp.dset import DemoDset
from lmp.tknzr import BPETknzr


def test_default_values() -> None:
  """Ensure default values consistency."""
  args = lmp.script.train_tknzr.parse_args(argv=[BPETknzr.tknzr_name])
  assert args.dset_name == DemoDset.dset_name
  assert args.exp_name == 'my_tknzr_exp'
  assert not args.is_uncased
  assert args.max_vocab == -1
  assert args.min_count == 0
  assert args.n_merge == 10000
  assert args.seed == 42
  assert args.ver == DemoDset.df_ver


def test_parse_results(
  exp_name: str,
  is_uncased: bool,
  max_vocab: int,
  min_count: int,
  n_merge: int,
  seed: int,
) -> None:
  """Must correctly parse all arguments for :py:class:`lmp.tknzr.BPETknzr`."""
  for dset_name, dset_type in lmp.dset.DSET_OPTS.items():
    for ver in dset_type.vers:
      argv = [
        BPETknzr.tknzr_name,
        '--dset_name',
        dset_name,
        '--exp_name',
        exp_name,
        '--max_vocab',
        str(max_vocab),
        '--min_count',
        str(min_count),
        '--n_merge',
        str(n_merge),
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
      assert args.max_vocab == max_vocab
      assert args.min_count == min_count
      assert args.n_merge == n_merge
      assert args.seed == seed
      assert args.tknzr_name == BPETknzr.tknzr_name
      assert args.ver == ver
