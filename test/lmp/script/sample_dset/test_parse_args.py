"""Test parsing arguments.

Test target:
- :py:meth:`lmp.script.sample_dset.parse_args`.
"""

import lmp.script.sample_dset
from lmp.dset import ALL_DSETS


def test_default_values() -> None:
  """Ensure default values consistency."""
  for dset_type in ALL_DSETS:
    args = lmp.script.sample_dset.parse_args(argv=[
      dset_type.dset_name,
    ])
    assert args.dset_name == dset_type.dset_name
    assert args.idx == 0
    assert args.ver == dset_type.df_ver


def test_parse_results() -> None:
  """Must correctly parse all arguments."""
  for dset_type in ALL_DSETS:
    for ver in dset_type.vers:
      for idx in range(0, 10):
        args = lmp.script.sample_dset.parse_args(argv=[
          dset_type.dset_name,
          '--idx',
          str(idx),
          '--ver',
          ver,
        ])
        assert args.dset_name == dset_type.dset_name
        assert args.idx == idx
        assert args.ver == ver
