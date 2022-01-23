"""Test the construction of :py:class:`lmp.infer.TopKInfer`.

Test target:
- :py:meth:`lmp.infer.TopKInfer.__init__`.
"""

from lmp.infer import TopKInfer


def test_init(k: int, max_seq_len: int) -> None:
  """Must correctly contruct inference method."""
  infer = TopKInfer(k=k, max_seq_len=max_seq_len)
  assert isinstance(infer, TopKInfer)
  assert infer.k == k
  assert infer.max_seq_len == max_seq_len
