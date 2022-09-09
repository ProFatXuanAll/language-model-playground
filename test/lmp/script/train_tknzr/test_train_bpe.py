"""Test training :py:class:`lmp.tknzr.BPETknzr`.

Test target:
- :py:meth:`lmp.script.train_tknzr.main`.
- :py:meth:`lmp.tknzr.BPETknzr.build_vocab`.
"""

import os

import lmp.script.train_tknzr
import lmp.util.cfg
import lmp.util.tknzr
from lmp.dset import WikiText2Dset
from lmp.tknzr import BPETknzr
from lmp.tknzr._bpe import EOW_TK


def test_train_bpe_tknzr_on_wiki_text_2(
  cfg_file_path: str,
  exp_name: str,
  is_uncased: bool,
  max_vocab: int,
  min_count: int,
  n_merge: int,
  seed: int,
  tknzr_file_path: str,
) -> None:
  """Must successfully train :py:class:`lmp.tknzr.BPETknzr` on :py:class:`lmp.dset.WikiText2Dset`."""
  argv = [
    BPETknzr.tknzr_name,
    '--dset_name',
    WikiText2Dset.dset_name,
    '--exp_name',
    exp_name,
    '--max_vocab',
    str(max_vocab),
    '--min_count',
    str(min_count),
    '--n_merge',
    str(n_merge),
    '--seed',
    str(seed),
    '--ver',
    'valid',  # Make training faster.
  ]

  if is_uncased:
    argv.append('--is_uncased')

  # Train tokenizer.
  lmp.script.train_tknzr.main(argv=argv)

  # Ensure configuration consistency.
  assert os.path.exists(cfg_file_path)
  assert lmp.util.cfg.load(exp_name=exp_name) == lmp.script.train_tknzr.parse_args(argv=argv)

  # Ensure tokenizer consistency.
  assert os.path.exists(tknzr_file_path)
  tknzr = lmp.util.tknzr.load(exp_name=exp_name)
  assert tknzr.is_uncased == is_uncased
  assert tknzr.max_vocab == max_vocab
  assert tknzr.min_count == min_count
  assert tknzr.n_merge == n_merge

  # Must include most frequent alphabets.
  assert f'the{EOW_TK}' in tknzr.tk2id or f'a{EOW_TK}' in tknzr.tk2id
