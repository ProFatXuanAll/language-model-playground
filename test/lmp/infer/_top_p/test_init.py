"""Test the construction of :py:class:`lmp.infer.TopPInfer`.

Test target:
- :py:meth:`lmp.infer.TopPInfer.__init__`.
"""

import math

from lmp.infer import TopPInfer


def test_init(max_seq_len: int, p: float) -> None:
  """Must correctly contruct inference method."""
  infer = TopPInfer(max_seq_len=max_seq_len, p=p)
  assert isinstance(infer, TopPInfer)
  assert infer.max_seq_len == max_seq_len
  assert math.isclose(infer.p, p)
