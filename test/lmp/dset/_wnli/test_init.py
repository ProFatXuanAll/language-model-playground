"""Test the construction of :py:class:`lmp.dset._wnli.WNLIDset`.

Test target:
- :py:meth:`lmp.dset._wnli.WNLIDset.__init__`.
"""

import lmp.dset._wnli


def test_default_version() -> None:
  """Must be able to construct the default version."""
  dset = lmp.dset._wnli.WNLIDset(ver=None)
  assert dset.ver == lmp.dset._wnli.WNLIDset.df_ver


def test_all_verions() -> None:
  """Must be able to construct all versions of :py:class:`lmp.dset._wnli.WNLIDset`."""
  for ver in lmp.dset._wnli.WNLIDset.vers:
    dset = lmp.dset._wnli.WNLIDset(ver=ver)
    assert dset.ver == ver
    assert len(dset) > 0
    assert all(map(lambda spl: isinstance(spl, str), dset))
    assert all(map(lambda spl: len(spl) > 0, dset))
