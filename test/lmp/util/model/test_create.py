r"""Test construction utilities for all language models.

Test target:
- :py:meth:`lmp.util.model.create`.
"""

import lmp.util.model
from lmp.model import GRUModel, LSTMModel, RNNModel
from lmp.tknzr import BaseTknzr


def test_create_rnn(tknzr: BaseTknzr):
    r"""Test construction for RNN language model."""
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

    assert isinstance(model, RNNModel), \
        '`model` must be an instance of `RNNModel`'
    assert model.emb.num_embeddings == tknzr.vocab_size, \
        '`model` must have correct numbers of embeddings.'
    assert model.emb.embedding_dim == d_emb, \
        '`model` must have correct embedding dimension.'
    assert model.emb.padding_idx == tknzr.pad_tkid, \
        '`model` must have correct padding token id.'
    assert model.emb_dp.p == p_emb, \
        '`model` must have correct embedding dropout probability.'
    assert model.pre_hid[0].out_features == d_hid, \
        '`model` must have correct hidden dimension'
    assert model.pre_hid[2].p == p_hid, \
        '`model` must have correct hidden dropout probability.'
    assert len(model.pre_hid) == n_pre_hid_lyr * 3, \
        '`model` must have correct number of pre-hidden layers.'
    assert model.hid.num_layers == n_hid_lyr, \
        '`model` must have correct number of hidden layers.'
    assert len(model.post_hid) == 3 * (n_post_hid_lyr - 1) + 2, \
        '`model` must have correct number of post-hidden layers.'


def test_create_gru(tknzr: BaseTknzr):
    r"""Test construction for GRU language model."""
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

    assert isinstance(model, GRUModel), \
        '`model` must be an instance of `GRUModel`'
    assert model.emb.num_embeddings == tknzr.vocab_size, \
        '`model` must have correct numbers of embeddings.'
    assert model.emb.embedding_dim == d_emb, \
        '`model` must have correct embedding dimension.'
    assert model.emb.padding_idx == tknzr.pad_tkid, \
        '`model` must have correct padding token id.'
    assert model.emb_dp.p == p_emb, \
        '`model` must have correct embedding dropout probability.'
    assert model.pre_hid[0].out_features == d_hid, \
        '`model` must have correct hidden dimension'
    assert model.pre_hid[2].p == p_hid, \
        '`model` must have correct hidden dropout probability.'
    assert len(model.pre_hid) == n_pre_hid_lyr * 3, \
        '`model` must have correct number of pre-hidden layers.'
    assert model.hid.num_layers == n_hid_lyr, \
        '`model` must have correct number of hidden layers.'
    assert len(model.post_hid) == 3 * (n_post_hid_lyr - 1) + 2, \
        '`model` must have correct number of post-hidden layers.'


def test_create_lstm(tknzr: BaseTknzr):
    r"""Test construction for LSTM language model."""
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

    assert isinstance(model, LSTMModel), \
        '`model` must be an instance of `GRUModel`'
    assert model.emb.num_embeddings == tknzr.vocab_size, \
        '`model` must have correct numbers of embeddings.'
    assert model.emb.embedding_dim == d_emb, \
        '`model` must have correct embedding dimension.'
    assert model.emb.padding_idx == tknzr.pad_tkid, \
        '`model` must have correct padding token id.'
    assert model.emb_dp.p == p_emb, \
        '`model` must have correct embedding dropout probability.'
    assert model.pre_hid[0].out_features == d_hid, \
        '`model` must have correct hidden dimension'
    assert model.pre_hid[2].p == p_hid, \
        '`model` must have correct hidden dropout probability.'
    assert len(model.pre_hid) == n_pre_hid_lyr * 3, \
        '`model` must have correct number of pre-hidden layers.'
    assert model.hid.num_layers == n_hid_lyr, \
        '`model` must have correct number of hidden layers.'
    assert len(model.post_hid) == 3 * (n_post_hid_lyr - 1) + 2, \
        '`model` must have correct number of post-hidden layers.'
