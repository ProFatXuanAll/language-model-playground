"""Test perplexity calculation result.

Test target:
- :py:meth:`lmp.script.eval_txt_ppl.main`.
"""

import math

import lmp.script.eval_txt_ppl


def test_ppl_output(capsys, ckpt: int, model_exp_name: str) -> None:
  """Must correctly output perplexity."""
  txt = 'Hello world'
  lmp.script.eval_txt_ppl.main(argv=[
    '--exp_name',
    model_exp_name,
    '--ckpt',
    str(ckpt),
    '--txt',
    txt,
  ])

  captured = capsys.readouterr()
  assert captured.out
  assert not math.isnan(float(captured.out))
