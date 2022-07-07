"""Test sample result.

Test target:
- :py:meth:`lmp.script.sample_dset.main`.
"""

import lmp.script.sample_dset
from lmp.dset import ChPoemDset, DemoDset, WikiText2Dset


def test_chinese_poem_sample(capsys) -> None:
  """Must correctly sample data points of :py:class:`lmp.dset.ChPoemDset`."""
  lmp.script.sample_dset.main(argv=[ChPoemDset.dset_name])

  captured = capsys.readouterr()
  assert '別路雲初起,離亭葉正飛。所嗟人異雁,不作一行歸。' in captured.out


def test_demo_sample(capsys) -> None:
  """Must correctly sample data points of :py:class:`lmp.dset.DemoDset`."""
  lmp.script.sample_dset.main(argv=[DemoDset.dset_name])

  captured = capsys.readouterr()
  assert 'If you add 0 to 1 you get 1 .' in captured.out


def test_wiki_text_2_sample(capsys) -> None:
  """Must correctly sample data points of :py:class:`lmp.dset.WikiText2Dset`."""
  lmp.script.sample_dset.main(argv=[WikiText2Dset.dset_name])

  captured = capsys.readouterr()
  assert 'Was it a vision , or a waking dream ?' in captured.out
