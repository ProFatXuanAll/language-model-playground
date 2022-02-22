"""Test top-1 generation with dummy model.

Test target:
- :py:meth:`lmp.infer.Top1Infer.gen`.
"""

from lmp.infer import Top1Infer
from lmp.model import BaseModel
from lmp.tknzr import BaseTknzr


def test_gen(gen_max_non_sp_tk_model: BaseModel, max_non_sp_tk: str, max_seq_len: int, tknzr: BaseTknzr) -> None:
  """Only generate ``max_non_sp_tk_model``."""
  infer = Top1Infer(max_seq_len=max_seq_len)

  expected = max_non_sp_tk * (max_seq_len - 1)

  out = infer.gen(model=gen_max_non_sp_tk_model, tknzr=tknzr, txt=max_non_sp_tk)

  assert out == expected
