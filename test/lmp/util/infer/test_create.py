"""Test construction utilities for all inference methods.

Test target:
- :py:meth:`lmp.util.infer.create`.
"""

import math

import lmp.util.infer
from lmp.infer import Top1Infer, TopKInfer, TopPInfer


def test_create_top_1(max_seq_len: int) -> None:
  """Test constuction for :py:class:`lmp.infer.Top1Infer`."""
  infer = lmp.util.infer.create(infer_name=Top1Infer.infer_name, max_seq_len=max_seq_len)
  assert isinstance(infer, Top1Infer)
  assert infer.max_seq_len == max_seq_len


def test_create_top_k(k: int, max_seq_len: int) -> None:
  """Capable of creating Top-k inference method."""
  top_k_infer = lmp.util.infer.create(infer_name=TopKInfer.infer_name, k=k, max_seq_len=max_seq_len)
  assert isinstance(top_k_infer, TopKInfer)
  assert top_k_infer.max_seq_len == max_seq_len
  assert top_k_infer.k == k


def test_create_top_p(max_seq_len: int, p: float) -> None:
  """Capable of creating Top-p inference method."""
  top_p_infer = lmp.util.infer.create(infer_name=TopPInfer.infer_name, max_seq_len=max_seq_len, p=p)
  assert isinstance(top_p_infer, TopPInfer)
  assert top_p_infer.max_seq_len == max_seq_len
  assert math.isclose(top_p_infer.p, p)
