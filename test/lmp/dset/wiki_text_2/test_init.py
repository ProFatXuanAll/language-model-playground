"""Test the construction of :py:class:`lmp.dset.WikiText2Dset`.

Test target:
- :py:meth:`lmp.dset.WikiText2Dset.__init__`.
"""

from typing import List

from lmp.dset import WikiText2Dset


def test_default_version(wiki_text_2_file_paths: List[str]) -> None:
  """Must be able to construct default version."""
  dset = WikiText2Dset(ver=None)
  assert dset.ver == WikiText2Dset.df_ver


def test_all_verions(wiki_text_2_file_paths: List[str]) -> None:
  """Must be able to construct all versions of :py:class:`lmp.dset.WikiText2Dset`."""
  for ver in WikiText2Dset.vers:
    dset = WikiText2Dset(ver=ver)
    assert dset.ver == ver
    assert len(dset) > 0
    assert all(map(lambda spl: isinstance(spl, str), dset))
    assert all(map(lambda spl: len(spl) > 0, dset))
