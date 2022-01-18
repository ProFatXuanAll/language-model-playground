"""Test save and load operation for tokenizer configuration.

Test target:
- :py:meth:`lmp.tknzr.WsTknzr.load`.
- :py:meth:`lmp.tknzr.WsTknzr.save`.
"""

import os
from typing import Dict

from lmp.tknzr import WsTknzr


def test_save_and_load(
  is_uncased: bool,
  exp_name: str,
  max_vocab: int,
  min_count: int,
  tk2id: Dict[str, int],
  tknzr_file_path: str,
) -> None:
  """Must correctly save and load tokenizer."""
  tknzr = WsTknzr(is_uncased=is_uncased, max_vocab=max_vocab, min_count=min_count, tk2id=tk2id)
  tknzr.save(exp_name=exp_name)

  assert os.path.exists(tknzr_file_path)

  load_tknzr = WsTknzr.load(exp_name)
  assert tknzr.__dict__ == load_tknzr.__dict__
