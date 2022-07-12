"""Test loading datasets.

Test target:
- :py:meth:`lmp.util.dset.load`.
"""

import lmp.util.dset
from lmp.dset import ChPoemDset, DemoDset, WNLIDset, WikiText2Dset


def test_load_chinese_poem() -> None:
  """Capable of loading Chinese poem datasets."""
  for ver in ChPoemDset.vers:
    dset = lmp.util.dset.load(dset_name=ChPoemDset.dset_name, ver=ver)
    assert isinstance(dset, ChPoemDset)
    assert dset.ver == ver


def test_load_demo() -> None:
  """Capable of loading Demo datasets."""
  for ver in DemoDset.vers:
    dset = lmp.util.dset.load(dset_name=DemoDset.dset_name, ver=ver)
    assert isinstance(dset, DemoDset)
    assert dset.ver == ver


def test_load_wnli() -> None:
  """Capable of loading WNLI datasets."""
  for ver in WNLIDset.vers:
    dset = lmp.util.dset.load(dset_name=WNLIDset.dset_name, ver=ver)
    assert isinstance(dset, WNLIDset)
    assert dset.ver == ver


def test_load_wikitext2() -> None:
  """Capable of loading Wiki-Text-2 datasets."""
  for ver in WikiText2Dset.vers:
    dset = lmp.util.dset.load(dset_name=WikiText2Dset.dset_name, ver=ver)
    assert isinstance(dset, WikiText2Dset)
    assert dset.ver == ver
