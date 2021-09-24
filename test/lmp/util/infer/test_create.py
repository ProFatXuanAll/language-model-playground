r"""Test save and load operation for training configuration.

Test target:
- :py:meth:`lmp.util.infer.create`.
"""

from lmp.infer import Top1Infer
import lmp.util.infer


def test_create() -> Top1Infer:
    r"""Ensure create the correct ``Top1Infer`` infer."""
    infer_name = 'top-1'
    max_seq_len = 1

    infer = lmp.util.infer.create(
        max_seq_len=max_seq_len,
        infer_name=infer_name,
    )

    isinstance(infer, Top1Infer)
    assert infer.infer_name == infer_name
    assert infer.max_seq_len == max_seq_len
