r"""Test the constructor of :py:class:`lmp.model._gru.GRUModel`.

Test target:
- :py:meth:`lmp.model._gru.GRUModel.__init__`.
"""
import torch.nn as nn


def test_emb(gru_model, tknzr, d_emb):
    r"""``emb`` must be an instance of `nn.Embedding` and ``emb`` must
    construct the right shape."""
    # Check the embedding type
    assert isinstance(gru_model.emb, nn.Embedding)

    # Check the shape of embedding
    assert gru_model.emb.num_embeddings == tknzr.vocab_size
    assert gru_model.emb.embedding_dim == d_emb
    assert gru_model.emb.padding_idx == tknzr.pad_tkid


def test_emb_dp(gru_model, p_emb):
    r"""``emb_dp`` must be an instance of `nn.Dropout` and ``emb_dp`` must
    be the right value."""
    # Check the embedding dropout type
    assert isinstance(gru_model.emb_dp, nn.Dropout)

    # Check the value of embedding dropout
    assert gru_model.emb_dp.p == p_emb


def test_pre_hid(gru_model, d_emb, p_hid, d_hid, n_pre_hid_lyr):
    r"""``pre_hid`` must be an instance of `nn.Sequential` and ``pre_hid`` must
    construct the right shape."""
    # Check the pre hidden embedding type
    assert isinstance(gru_model.pre_hid, nn.Sequential)
    assert isinstance(gru_model.pre_hid[1], nn.ReLU)

    # Check the shape of pre hidden embedding
    assert gru_model.pre_hid[0].in_features == d_emb
    assert gru_model.pre_hid[0].out_features == d_hid
    assert gru_model.pre_hid[2].p == p_hid

    for i in range(0, n_pre_hid_lyr, 3):
        # Check the pre hidden layer parameters
        assert gru_model.pre_hid[i + 3].in_features == d_hid
        assert gru_model.pre_hid[i + 3].out_features == d_hid
        assert gru_model.pre_hid[i + 5].p == p_hid


def test_hid(gru_model, n_hid_lyr, d_hid, p_hid):
    r"""``hid`` must be an instance of `nn.GRU` and ``hid`` must
    construct the right shape."""
    # Check the type of hidden layer
    assert isinstance(gru_model.hid, nn.GRU)

    # Check the value of hidden layer
    if n_hid_lyr == 1:
        assert gru_model.hid.input_size == d_hid
        assert gru_model.hid.hidden_size == d_hid
        assert gru_model.hid.batch_first
    else:
        assert gru_model.hid.input_size == d_hid
        assert gru_model.hid.hidden_size == d_hid
        assert gru_model.hid.batch_first
        assert gru_model.hid.dropout == p_hid


def test_post_hid(gru_model, d_emb, p_hid, d_hid, n_pre_hid_lyr):
    r"""``post_hid`` must be an instance of `nn.Sequential` and ``post_hid`` must
    construct the right shape."""
    # Check the post hidden embedding type
    assert isinstance(gru_model.post_hid, nn.Sequential)

    # Check the shape of post hidden embedding
    for i in range(0, n_pre_hid_lyr, 3):
        # Check the post hidden layer parameters
        assert gru_model.post_hid[i].p == p_hid
        assert gru_model.post_hid[i + 1].in_features == d_hid
        assert gru_model.post_hid[i + 1].out_features == d_hid

        # Check the post_hid layer type
        assert isinstance(gru_model.post_hid[i + 2], nn.ReLU)

    # Check the post hidden last two layer parameters
    assert gru_model.post_hid[-1].in_features == d_hid
    assert gru_model.post_hid[-1].out_features == d_emb
    assert gru_model.post_hid[-2].p == p_hid
