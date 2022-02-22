"""Test top-k generation with dummy model.

Test target:
- :py:meth:`lmp.infer.TopKInfer.gen`.
"""

from collections import Counter

from lmp.infer import TopKInfer
from lmp.model import BaseModel
from lmp.tknzr import BaseTknzr


def test_gen(
  gen_max_non_sp_tk_model: BaseModel,
  k: int,
  max_non_sp_tk: str,
  max_seq_len: int,
  tknzr: BaseTknzr,
) -> None:
  """Only generate ``max_non_sp_tk_model``."""
  infer = TopKInfer(k=k, max_seq_len=max_seq_len)
  counter = Counter()

  for _ in range(100):
    out = infer.gen(model=gen_max_non_sp_tk_model, tknzr=tknzr, txt=max_non_sp_tk)
    counter.update(out)

  # The probability of failing this assertion must be extremely low.
  assert counter.most_common()[0][0] == max_non_sp_tk
