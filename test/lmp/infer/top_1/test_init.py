"""Test the construction of :py:class:`lmp.infer.Top1Infer`.

Test target:
- :py:meth:`lmp.infer.Top1Infer.__init__`.
"""

from lmp.infer import Top1Infer


def test_init(max_seq_len: int) -> None:
  """Must correctly contruct inference method."""
  infer = Top1Infer(max_seq_len=max_seq_len)
  assert isinstance(infer, Top1Infer)
  assert infer.max_seq_len == max_seq_len
