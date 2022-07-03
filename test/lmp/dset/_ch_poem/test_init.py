"""Test the construction of :py:class:`lmp.dset._ch_poem.ChPoemDset`.

Test target:
- :py:meth:`lmp.dset._ch_poem.ChPoemDset.__init__`.
"""

import lmp.dset._ch_poem


def test_default_version() -> None:
  """Must be able to construct the default version."""
  dset = lmp.dset._ch_poem.ChPoemDset(ver=None)
  assert dset.ver == lmp.dset._ch_poem.ChPoemDset.df_ver


def test_all_verions() -> None:
  """Must be able to construct all versions of :py:class:`lmp.dset._ch_poem.ChPoemDset`."""
  for ver in lmp.dset._ch_poem.ChPoemDset.vers:
    dset = lmp.dset._ch_poem.ChPoemDset(ver=ver)
    assert dset.ver == ver
    assert len(dset) > 0
    assert all(map(lambda spl: isinstance(spl, str), dset))
    assert all(map(lambda spl: len(spl) > 0, dset))
