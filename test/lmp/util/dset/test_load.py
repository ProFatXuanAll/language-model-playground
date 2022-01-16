"""Test loading datasets.

Test target:
- :py:meth:`lmp.util.dset.load`.
"""

import lmp.util.dset
from lmp.dset import ChPoemDset, WikiText2Dset


def test_load_wikitext2():
  """Capable of loading Wiki-Text-2 datasets."""

  for ver in WikiText2Dset.vers:
    dset = lmp.util.dset.load(
      dset_name=WikiText2Dset.dset_name,
      ver=ver,
    )

    assert isinstance(dset, WikiText2Dset)
    assert dset.ver == ver


def test_load_chinese_poem():
  """Capable of loading Chinese poem datasets."""

  for ver in ChPoemDset.vers:
    dset = lmp.util.dset.load(
      dset_name=ChPoemDset.dset_name,
      ver=ver,
    )

    assert isinstance(dset, ChPoemDset)
    assert dset.ver == ver
