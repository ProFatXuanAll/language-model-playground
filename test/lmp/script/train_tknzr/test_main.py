"""Test training tokenizer script.

Test target:
- :py:meth:`lmp.script.train_tknzr.main`.
"""

import os
from typing import List

import lmp.script.train_tknzr
import lmp.util.cfg
import lmp.util.tknzr
from lmp.dset import WikiText2Dset
from lmp.tknzr import CharTknzr


def test_train_char_tknzr_on_wiki_text_2(
  cfg_file_path: str,
  exp_name: str,
  is_uncased: bool,
  max_vocab: int,
  min_count: int,
  wiki_text_2_file_paths: List[str],
  tknzr_file_path: str,
) -> None:
  """Successfully train tokenizer :py:class:`lmp.tknzr.CharTknzr` on :py:class:`lmp.dset.WikiText2Dset` dataset."""
  argv = [
    CharTknzr.tknzr_name,
    '--dset_name',
    WikiText2Dset.dset_name,
    '--exp_name',
    exp_name,
    '--max_vocab',
    str(max_vocab),
    '--min_count',
    str(min_count),
    '--ver',
    WikiText2Dset.df_ver,
  ]
  if is_uncased:
    argv.append('--is_uncased')

  lmp.script.train_tknzr.main(argv=argv)
  assert os.path.exists(cfg_file_path)
  assert os.path.exists(tknzr_file_path)

  cfg = lmp.util.cfg.load(exp_name=exp_name)
  assert cfg.dset_name == WikiText2Dset.dset_name
  assert cfg.exp_name == exp_name
  assert cfg.is_uncased == is_uncased
  assert cfg.max_vocab == max_vocab
  assert cfg.min_count == min_count
  assert cfg.ver == WikiText2Dset.df_ver

  tknzr = lmp.util.tknzr.load(exp_name=exp_name, tknzr_name=CharTknzr.tknzr_name)
  assert tknzr.is_uncased == is_uncased
  assert tknzr.max_vocab == max_vocab
  assert tknzr.min_count == min_count
  # Must include most frequent alphabets.
  assert 't' in tknzr.tk2id
  assert 'h' in tknzr.tk2id
  assert 'e' in tknzr.tk2id
