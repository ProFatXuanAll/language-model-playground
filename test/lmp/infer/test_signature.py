"""Test :py:mod:`lmp.infer` signatures."""

import lmp.infer


def test_module_attribute() -> None:
  """Ensure module attributes' signatures."""
  assert hasattr(lmp.infer, 'BaseInfer')
  assert hasattr(lmp.infer, 'Top1Infer')
  assert hasattr(lmp.infer, 'TopKInfer')
  assert hasattr(lmp.infer, 'TopPInfer')
  assert hasattr(lmp.infer, 'ALL_INFERS')
  assert lmp.infer.ALL_INFERS == [
    lmp.infer.Top1Infer,
    lmp.infer.TopKInfer,
    lmp.infer.TopPInfer,
  ]
  assert hasattr(lmp.infer, 'INFER_OPTS')
  assert lmp.infer.INFER_OPTS == {
    lmp.infer.Top1Infer.infer_name: lmp.infer.Top1Infer,
    lmp.infer.TopKInfer.infer_name: lmp.infer.TopKInfer,
    lmp.infer.TopPInfer.infer_name: lmp.infer.TopPInfer,
  }
