r"""Test loading utilities for all language models.

Test target:
- :py:meth:`lmp.util.model.load`.
"""

import torch

import lmp.util.model
from lmp.model import GRUModel, LSTMModel, RNNModel
from lmp.tknzr import BaseTknzr


def test_load_rnn(exp_name: str, tknzr: BaseTknzr, clean_model):
    r"""Load back pre-trained RNN language model."""
    ckpt = 1
    d_emb = 1
    d_hid = 2
    n_hid_lyr = 1
    n_post_hid_lyr = 2
    n_pre_hid_lyr = 3
    p_emb = 0.1
    p_hid = 0.2

    model = lmp.util.model.create(
        model_name=RNNModel.model_name,
        d_emb=d_emb,
        d_hid=d_hid,
        n_hid_lyr=n_hid_lyr,
        n_post_hid_lyr=n_post_hid_lyr,
        n_pre_hid_lyr=n_pre_hid_lyr,
        tknzr=tknzr,
        p_emb=p_emb,
        p_hid=p_hid,
    )

    model.save(ckpt=ckpt, exp_name=exp_name)

    load_model = lmp.util.model.load(
        ckpt=ckpt,
        exp_name=exp_name,
        model_name=RNNModel.model_name,
        d_emb=d_emb,
        d_hid=d_hid,
        n_hid_lyr=n_hid_lyr,
        n_post_hid_lyr=n_post_hid_lyr,
        n_pre_hid_lyr=n_pre_hid_lyr,
        tknzr=tknzr,
        p_emb=p_emb,
        p_hid=p_hid,
    )

    # Test Case: Type check.
    assert isinstance(load_model, RNNModel)

    # Test Case: Parameters check.
    for (p_1, p_2) in zip(load_model.parameters(), model.parameters()):
        assert torch.equal(p_1, p_2)


def test_load_gru(exp_name: str, tknzr: BaseTknzr, clean_model):
    r"""Load back pre-trained GRU language model."""
    ckpt = 1
    d_emb = 1
    d_hid = 2
    n_hid_lyr = 1
    n_post_hid_lyr = 2
    n_pre_hid_lyr = 3
    p_emb = 0.1
    p_hid = 0.2

    model = lmp.util.model.create(
        model_name=GRUModel.model_name,
        d_emb=d_emb,
        d_hid=d_hid,
        n_hid_lyr=n_hid_lyr,
        n_post_hid_lyr=n_post_hid_lyr,
        n_pre_hid_lyr=n_pre_hid_lyr,
        tknzr=tknzr,
        p_emb=p_emb,
        p_hid=p_hid,
    )

    model.save(ckpt=ckpt, exp_name=exp_name)

    load_model = lmp.util.model.load(
        ckpt=ckpt,
        exp_name=exp_name,
        model_name=GRUModel.model_name,
        d_emb=d_emb,
        d_hid=d_hid,
        n_hid_lyr=n_hid_lyr,
        n_post_hid_lyr=n_post_hid_lyr,
        n_pre_hid_lyr=n_pre_hid_lyr,
        tknzr=tknzr,
        p_emb=p_emb,
        p_hid=p_hid,
    )

    # Test Case: Type check.
    assert isinstance(load_model, GRUModel)

    # Test Case: Parameters check.
    for (p_1, p_2) in zip(load_model.parameters(), model.parameters()):
        assert torch.equal(p_1, p_2)


def test_load_lstm(exp_name: str, tknzr: BaseTknzr, clean_model):
    r"""Load back pre-trained LSTM language model."""
    ckpt = 1
    d_emb = 1
    d_hid = 2
    n_hid_lyr = 1
    n_post_hid_lyr = 2
    n_pre_hid_lyr = 3
    p_emb = 0.1
    p_hid = 0.2

    model = lmp.util.model.create(
        model_name=LSTMModel.model_name,
        d_emb=d_emb,
        d_hid=d_hid,
        n_hid_lyr=n_hid_lyr,
        n_post_hid_lyr=n_post_hid_lyr,
        n_pre_hid_lyr=n_pre_hid_lyr,
        tknzr=tknzr,
        p_emb=p_emb,
        p_hid=p_hid,
    )

    model.save(ckpt=ckpt, exp_name=exp_name)

    load_model = lmp.util.model.load(
        ckpt=ckpt,
        exp_name=exp_name,
        model_name=LSTMModel.model_name,
        d_emb=d_emb,
        d_hid=d_hid,
        n_hid_lyr=n_hid_lyr,
        n_post_hid_lyr=n_post_hid_lyr,
        n_pre_hid_lyr=n_pre_hid_lyr,
        tknzr=tknzr,
        p_emb=p_emb,
        p_hid=p_hid,
    )

    # Test Case: Type check.
    assert isinstance(load_model, LSTMModel)

    # Test Case: Parameters check.
    for (p_1, p_2) in zip(load_model.parameters(), model.parameters()):
        assert torch.equal(p_1, p_2)
