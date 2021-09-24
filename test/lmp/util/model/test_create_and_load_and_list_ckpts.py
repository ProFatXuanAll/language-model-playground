r"""Test operation for new language model.

Test target:
- :py:meth:`lmp.util.model.create`.
- :py:meth:`lmp.util.model.load`.
- :py:meth:`lmp.util.model.list_ckpts`.
"""

import torch

from lmp.model import RNNModel
import lmp.util.model


def test_create(
    tknzr,
    exp_name,
):
    r"""Test creation for language model."""
    # Test Case: Create model.
    model = lmp.util.model.create(
        model_name='RNN',
        d_emb=10,
        d_hid=10,
        n_hid_lyr=2,
        n_post_hid_lyr=2,
        n_pre_hid_lyr=2,
        tknzr=tknzr,
        p_emb=0.1,
        p_hid=0.1,
    )

    isinstance(model, RNNModel)


def test_laod(
    tknzr,
    exp_name,
    clean_model,
):
    r"""Test loading for language model."""
    model = lmp.util.model.create(
        model_name='RNN',
        d_emb=10,
        d_hid=10,
        n_hid_lyr=2,
        n_post_hid_lyr=2,
        n_pre_hid_lyr=2,
        tknzr=tknzr,
        p_emb=0.1,
        p_hid=0.1,
    )

    # Test Case:  Load model.
    model.save(
        ckpt=1,
        exp_name=exp_name,
    )

    load_model = lmp.util.model.load(
        ckpt=1,
        exp_name=exp_name,
        model_name='RNN',
        d_emb=10,
        d_hid=10,
        n_hid_lyr=2,
        n_post_hid_lyr=2,
        n_pre_hid_lyr=2,
        tknzr=tknzr,
        p_emb=0.1,
        p_hid=0.1,
    )

    # Test Case: Type check.
    assert isinstance(load_model, RNNModel)

    # Test Case: Parameters check.
    for (p_1, p_2) in zip(load_model.parameters(), model.parameters()):
        assert torch.equal(p_1, p_2)

    # Test Case: Weight check.
    assert (load_model.emb.weight == model.emb.weight).all().item()

    # Test Case: List check.
    # Add another model to check list operation.
    model.save(
        ckpt=3,
        exp_name=exp_name,
    )

    load_ckpts = lmp.util.model.list_ckpts(
        exp_name=exp_name,
        first_ckpt=1,
        last_ckpt=3,
    )

    return load_ckpts == [1, 3]


def test_list_ckpts(
    tknzr,
    exp_name,
    clean_model,
):
    r"""Test operation for loading multiple language model."""
    # Test Case: Create model.
    model = lmp.util.model.create(
        model_name='RNN',
        d_emb=10,
        d_hid=10,
        n_hid_lyr=2,
        n_post_hid_lyr=2,
        n_pre_hid_lyr=2,
        tknzr=tknzr,
        p_emb=0.1,
        p_hid=0.1,
    )

    # Save multiple model's checkpoints.
    model.save(
        ckpt=1,
        exp_name=exp_name,
    )

    model.save(
        ckpt=3,
        exp_name=exp_name,
    )

    model.save(
        ckpt=5,
        exp_name=exp_name,
    )

    # Test Case: Loading the assigning range.
    load_ckpts = lmp.util.model.list_ckpts(
        exp_name=exp_name,
        first_ckpt=1,
        last_ckpt=3,
    )

    assert load_ckpts == [1, 3]

    # Test Case: Loading the last checkpoint.
    load_ckpts = lmp.util.model.list_ckpts(
        exp_name=exp_name,
        first_ckpt=-1,
        last_ckpt=5,
    )

    assert load_ckpts == [5]

    # Test Case: Loading all checkpoint.
    load_ckpts = lmp.util.model.list_ckpts(
        exp_name=exp_name,
        first_ckpt=1,
        last_ckpt=-1,
    )

    assert load_ckpts == [1, 3, 5]
