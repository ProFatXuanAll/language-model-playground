"""Test perplexity calculation result.

Test target:
- :py:meth:`lmp.script.eval_dset_ppl.main`.
"""

import math
import os
import re
from typing import List

import lmp.script.eval_dset_ppl
from lmp.dset import WikiText2Dset


def test_ppl_output(batch_size: int, capsys, ckpts: List[int], log_dir_path: str, model_exp_name: str) -> None:
  """Must correctly output perplexity."""
  lmp.script.eval_dset_ppl.main(
    argv=[
      WikiText2Dset.dset_name,
      '--batch_size',
      str(batch_size),
      '--exp_name',
      model_exp_name,
      '--first_ckpt',
      str(min(ckpts)),
      '--last_ckpt',
      str(max(ckpts)),
      '--ver',
      'valid',
    ]
  )

  assert os.path.exists(log_dir_path)

  captured = capsys.readouterr()
  assert captured.err

  for line in re.split(r'\n', captured.out):
    if not line:
      continue
    match = re.match(r'checkpoint: (\d+), avg ppl: (\d*.?\d*)', line)
    assert match
    assert int(match.group(1)) in ckpts
    assert not math.isnan(float(match.group(2)))
