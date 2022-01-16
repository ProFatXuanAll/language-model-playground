"""Test top-1 generation with dummy model.

Test target:
- :py:meth:`lmp.infer.TopKInfer.gen`.
"""

from lmp.infer import TopKInfer
from lmp.model import BaseModel
from lmp.tknzr import BaseTknzr


def test_gen(model: BaseModel, tknzr: BaseTknzr):
  max_seq_len = 10
  k = tknzr.vocab_size

  infer = TopKInfer(k=k, max_seq_len=max_seq_len)

  largest_tkid = max(tknzr.id2tk.keys())
  largest_tk = tknzr.id2tk[largest_tkid]

  # Dummy model will always generate largest token id as prediction result.
  # Thus we expect top-k inference method to output `max_seq_len - 1` tokens,
  # and all of which are `largest_tk`.
  # Minus 1 is required since tokenizer will prepend `[bos]` at front.
  expected = largest_tk * (max_seq_len - 1)

  out = infer.gen(
    model=model,
    tknzr=tknzr,
    txt=largest_tk,
  )

  assert out == expected
