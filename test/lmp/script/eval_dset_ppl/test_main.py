"""Test perplexity calculation result.

Test target:
- :py:meth:`lmp.script.eval_dset_ppl.main`.
"""

import math
import os
import re
from typing import List

import lmp.script.eval_dset_ppl
from lmp.dset import DemoDset


def test_ppl_output(
  batch_size: int,
  capsys,
  ckpts: List[int],
  eval_log_dir_path: str,
  model_exp_name: str,
  seed: int,
) -> None:
  """Must correctly output perplexity."""
  argv = [
    DemoDset.dset_name,
    '--batch_size',
    str(batch_size),
    '--exp_name',
    model_exp_name,
    '--first_ckpt',
    str(min(ckpts)),
    '--last_ckpt',
    str(max(ckpts)),
    '--seed',
    str(seed),
    '--ver',
    'valid',
  ]

  lmp.script.eval_dset_ppl.main(argv=argv)

  assert os.path.exists(eval_log_dir_path)

  captured = capsys.readouterr()
  assert captured.err

  ppls = []
  for line in re.split(r'\n', captured.out):
    if not line:
      continue

    match_1 = re.match(r'checkpoint: (\d+), ppl: (\d*.?\d*)', line)
    match_2 = re.match(r'best checkpoint: (\d+), best ppl: (\d*.?\d*)', line)
    assert match_1 or match_2

    if match_1:
      assert int(match_1.group(1)) in ckpts
      assert not math.isnan(float(match_1.group(2)))
      ppls.append(match_1.group(2))
    if match_2:
      assert int(match_2.group(1)) in ckpts
      assert match_2.group(2) in ppls
