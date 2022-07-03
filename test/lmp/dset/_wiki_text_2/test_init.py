"""Test the construction of :py:class:`lmp.dset._wiki_text_2.WikiText2Dset`.

Test target:
- :py:meth:`lmp.dset._wiki_text_2.WikiText2Dset.__init__`.
"""

import lmp.dset._wiki_text_2


def test_default_version() -> None:
  """Must be able to construct the default version."""
  dset = lmp.dset._wiki_text_2.WikiText2Dset(ver=None)
  assert dset.ver == lmp.dset._wiki_text_2.WikiText2Dset.df_ver


def test_all_verions() -> None:
  """Must be able to construct all versions of :py:class:`lmp.dset._wiki_text_2.WikiText2Dset`."""
  for ver in lmp.dset._wiki_text_2.WikiText2Dset.vers:
    dset = lmp.dset._wiki_text_2.WikiText2Dset(ver=ver)
    assert dset.ver == ver
    assert len(dset) > 0
    assert all(map(lambda spl: isinstance(spl, str), dset))
    assert all(map(lambda spl: len(spl) > 0, dset))
