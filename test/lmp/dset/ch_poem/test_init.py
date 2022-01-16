"""Test the construction of :py:class:`lmp.dset.ChPoemDset`.

Test target:
- :py:meth:`lmp.dset.ChPoemDset.__init__`.
"""

from typing import List

from lmp.dset import ChPoemDset


def test_default_version(ch_poem_file_paths: List[str]) -> None:
  """Must be able to construct default version."""
  dset = ChPoemDset(ver=None)
  assert dset.ver == ChPoemDset.df_ver


def test_all_verions(ch_poem_file_paths: List[str]) -> None:
  """Must be able to construct all versions of :py:class:`lmp.dset.ChPoemDset`."""
  for ver in ChPoemDset.vers:
    dset = ChPoemDset(ver=ver)
    assert dset.ver == ver
    assert len(dset) > 0
    assert all(map(lambda spl: isinstance(spl, str), dset))
    assert all(map(lambda spl: len(spl) > 0, dset))
