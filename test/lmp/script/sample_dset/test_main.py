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
  assert '風淅淅。夜雨連雲黑。滴滴。窗外芭蕉燈下客。除非魂夢到鄉國。免被關山隔。憶憶。一句枕前爭忘得。' in captured.out


def test_demo_sample(capsys) -> None:
  """Must correctly sample data points of :py:class:`lmp.dset.DemoDset`."""
  lmp.script.sample_dset.main(argv=[DemoDset.dset_name])

  captured = capsys.readouterr()
  assert 'If you add 0 to 1 you get 1 .' in captured.out


def test_wiki_text_2_sample(capsys) -> None:
  """Must correctly sample data points of :py:class:`lmp.dset.WikiText2Dset`."""
  lmp.script.sample_dset.main(argv=[WikiText2Dset.dset_name])

  captured = capsys.readouterr()
  assert '= Valkyria Chronicles III =' in captured.out
