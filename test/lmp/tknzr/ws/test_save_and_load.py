"""Test save and load operation for tokenizer configuration.

Test target:
- :py:meth:`lmp.tknzr.WsTknzr.load`.
- :py:meth:`lmp.tknzr.WsTknzr.save`.
"""

import os
from typing import Dict

from lmp.tknzr import WsTknzr


def test_save_file_exist(
  is_uncased: bool,
  exp_name: str,
  max_vocab: int,
  min_count: int,
  tk2id: Dict[str, int],
  tknzr_file_path: str,
) -> None:
  """Save tokenizer as a file."""
  tknzr = WsTknzr(is_uncased=is_uncased, max_vocab=max_vocab, min_count=min_count, tk2id=tk2id)
  tknzr.save(exp_name=exp_name)

  assert os.path.exists(tknzr_file_path)


def test_load_result(
  is_uncased: bool,
  exp_name: str,
  max_vocab: int,
  min_count: int,
  tk2id: Dict[str, int],
  tknzr_file_path: str,
) -> None:
  """Ensure tokenizer consistency between save and load."""
  tknzr = WsTknzr(is_uncased=is_uncased, max_vocab=max_vocab, min_count=min_count, tk2id=tk2id)
  tknzr.save(exp_name=exp_name)

  load_tknzr = WsTknzr.load(exp_name)
  assert tknzr.__dict__ == load_tknzr.__dict__
