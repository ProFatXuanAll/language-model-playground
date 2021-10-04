r"""Test model checkpoints listing utilities.

Test target:
- :py:meth:`lmp.util.model.list_ckpts`.
"""

import lmp.util.model
from lmp.model import RNNModel
from lmp.tknzr import BaseTknzr


def test_list_ckpts(
    exp_name: str,
    tknzr: BaseTknzr,
    clean_model,
):
    r"""List specified language model checkpoints."""
    # Test Case: Create model.
    model = lmp.util.model.create(
        model_name=RNNModel.model_name,
        d_emb=1,
        d_hid=1,
        n_hid_lyr=1,
        n_post_hid_lyr=1,
        n_pre_hid_lyr=1,
        tknzr=tknzr,
        p_emb=0.1,
        p_hid=0.1,
    )

    # Save multiple model's checkpoints.
    model.save(ckpt=1, exp_name=exp_name)
    model.save(ckpt=3, exp_name=exp_name)
    model.save(ckpt=5, exp_name=exp_name)

    # Test Case: List checkpoints in the specified range.
    load_ckpts = lmp.util.model.list_ckpts(
        exp_name=exp_name,
        first_ckpt=1,
        last_ckpt=3,
    )

    assert load_ckpts == [1, 3]

    # Test Case: List only the last checkpoint.
    load_ckpts = lmp.util.model.list_ckpts(
        exp_name=exp_name,
        first_ckpt=-1,
        last_ckpt=5,
    )

    assert load_ckpts == [5]

    # Test Case: List all checkpoints.
    load_ckpts = lmp.util.model.list_ckpts(
        exp_name=exp_name,
        first_ckpt=1,
        last_ckpt=-1,
    )

    assert load_ckpts == [1, 3, 5]
