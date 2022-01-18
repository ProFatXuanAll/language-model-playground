"""Test sample result.

Test target:
- :py:meth:`lmp.script.sample_dset.main`.
"""

from typing import List

import lmp.script.sample_dset
from lmp.dset import ChPoemDset, WikiText2Dset


def test_chinese_poem_sample(capsys, ch_poem_file_paths: List[str]) -> None:
  """Must correctly sample data points of :py:class:`lmp.dset.ChPoemDset`."""
  lmp.script.sample_dset.main(argv=[ChPoemDset.dset_name])

  captured = capsys.readouterr()
  assert '風淅淅。夜雨連雲黑。滴滴。窗外芭蕉燈下客。除非魂夢到鄉國。免被關山隔。憶憶。一句枕前爭忘得。' in captured.out


def test_wiki_text_2_sample(capsys, wiki_text_2_file_paths: List[str]) -> None:
  """Must correctly sample data points of :py:class:`lmp.dset.WikiText2Dset`."""
  lmp.script.sample_dset.main(argv=[WikiText2Dset.dset_name])

  captured = capsys.readouterr()
  assert 'Senjō no Valkyria 3 : [unk] Chronicles ( Japanese : 戦場のヴァルキュリア3 , lit .' in captured.out
  assert 'Valkyria of the Battlefield 3 ) , commonly referred to as Valkyria Chronicles' in captured.out
  assert 'III outside Japan' in captured.out
